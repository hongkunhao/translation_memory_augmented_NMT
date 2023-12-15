# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyWithRetrievalLossSoftLabelCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def compute_retrieval_loss(target, TM_num_copy_seqs, input_for_learnable_p, ignore_index):
    # target: B x T -> B*T
    # TM_num_copy_seqs: list[] TM_num * TM_T x B tensor
    # input_for_learnable_p: B x T x TM_num
    batch_size, T, TM_num = input_for_learnable_p.size()
    TM_T, TM_bsz = TM_num_copy_seqs[0].size()
    assert TM_num == len(TM_num_copy_seqs)
    target = target.view(batch_size, T)  # B x T
    TM_num_copy_seqs_new = []
    for cur_copy_seqs in TM_num_copy_seqs:  # list[] TM_num * B x TM_T tensor
        cur_copy_seqs = cur_copy_seqs.transpose(0, 1)
        TM_num_copy_seqs_new.append(cur_copy_seqs)
    TM_num_copy_seqs = TM_num_copy_seqs_new
    for cur_copy_seqs in TM_num_copy_seqs:
        cur_B, cur_T = cur_copy_seqs.size()
        assert cur_B == batch_size
        assert cur_T == TM_T
    # retrieval_loss_target = torch.zeros_like(input_for_learnable_p)  # B x T x TM_num
    # retrieval_loss_target = retrieval_loss_target.float()
    retrieval_loss_target = []
    for cur_T_idx in range(T):
        cur_target = target[:, cur_T_idx]
        cur_target = cur_target.unsqueeze(-1)
        existance_num = torch.zeros_like(cur_target)  # B x 1
        existance_num = existance_num.float()
        existance_num = existance_num + 1e-45
        cur_target = cur_target.expand(batch_size, TM_T)  # B x TM_T
        cur_T_idx_loss_target = []
        for cur_TM_idx in range(TM_num):
            cur_copy_seqs = TM_num_copy_seqs[cur_TM_idx]
            cur_equal = cur_copy_seqs.eq(cur_target)
            cur_equal = torch.sum(cur_equal, dim=1, keepdim=True)  # B x 1
            if_exist = cur_equal.ge(1)  # B x 1
            existance_num = existance_num + if_exist
            if_exist = if_exist.float()
            cur_T_idx_loss_target.append(if_exist)
        cur_T_idx_loss_target = torch.cat(cur_T_idx_loss_target, dim=1)  # B x TM_num
        cur_T_idx_loss_target = torch.div(cur_T_idx_loss_target, existance_num)  # B x TM_num
        retrieval_loss_target.append(cur_T_idx_loss_target)
    retrieval_loss_target = torch.stack(retrieval_loss_target, dim=1)
    # retrieval_loss = torch.nn.functional.binary_cross_entropy(input_for_learnable_p, retrieval_loss_target, reduction='sum')
    #print(input_for_learnable_p)
    #print(input_for_learnable_p.size())
    #print(retrieval_loss_target)
    #print(retrieval_loss_target.size())
    retrieval_loss = torch.nn.functional.binary_cross_entropy(input=input_for_learnable_p, target=retrieval_loss_target, reduce=False)
    #print(retrieval_loss)
    pad_mask = target.eq(ignore_index)
    pad_mask = pad_mask[:, :, None]
    pad_mask = pad_mask.expand(batch_size, T, TM_num)
    retrieval_loss.masked_fill_(pad_mask, 0.0)
    # retrieval_loss = retrieval_loss.mean()
    retrieval_loss = retrieval_loss.sum()
    #print(retrieval_loss)
    return retrieval_loss


@register_criterion(
    "label_smoothed_cross_entropy_with_retrieval_loss_soft_label", dataclass=LabelSmoothedCrossEntropyWithRetrievalLossSoftLabelCriterionConfig
)
class LabelSmoothedCrossEntropyWithRetrievalLossSoftLabelCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, retrieval_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "retrieval_loss": retrieval_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        TM_num_copy_seqs = net_output[2]  # list[] TM_num T x B tensor
        input_for_learnable_p = net_output[3]  # B x T x TM_num
        retrieval_loss = compute_retrieval_loss(target=target, TM_num_copy_seqs=TM_num_copy_seqs, input_for_learnable_p=input_for_learnable_p, ignore_index=self.padding_idx)
        loss = loss + 0.1 * retrieval_loss
        return loss, nll_loss, retrieval_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        retrieval_loss_sum = sum(log.get("retrieval_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "retrieval_loss", retrieval_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
