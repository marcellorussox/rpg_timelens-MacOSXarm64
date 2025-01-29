import torch as th
import torch.nn.functional as F
from timelens import refine_warp_network, warp_network
from timelens.superslomo import unet


def _pack_input_for_attention_computation(example):
    fusion = example["middle"]["fusion"]
    number_of_examples, _, height, width = fusion.size()

    # Controlla se `weight` è una lista e convertila in un tensore
    if isinstance(example["middle"]["weight"], list):
        weight = th.tensor(example["middle"]["weight"], dtype=th.float32, device=fusion.device)
    else:
        weight = example["middle"]["weight"].to(dtype=th.float32, device=fusion.device)

    # Prendi il device del primo tensore (fusion) e sposta tutto su di esso
    device = fusion.device
    tensors = [
        example["after"]["flow"].to(device),
        example["middle"]["after_refined_warped"].to(device),
        example["before"]["flow"].to(device),
        example["middle"]["before_refined_warped"].to(device),
        example["middle"]["fusion"].to(device),
        weight.view(-1, 1, 1, 1).expand(number_of_examples, 1, height, width).type(fusion.dtype),
    ]

    return th.cat(tensors, dim=1)


def _compute_weighted_average(attention, before_refined, after_refined, fusion):
    # Assicura che tutti i tensori siano sullo stesso device
    device = attention.device
    before_refined = before_refined.to(device)
    after_refined = after_refined.to(device)

    return (
            attention[:, 0, ...].unsqueeze(1) * before_refined
            + attention[:, 1, ...].unsqueeze(1) * after_refined
            + attention[:, 2, ...].unsqueeze(1) * fusion
    )


class AttentionAverage(refine_warp_network.RefineWarp):
    def __init__(self):
        warp_network.Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)
        self.attention_network = unet.UNet(14, 3, False)

    def run_fast(self, example):
        example['middle']['before_refined_warped'], \
            example['middle']['after_refined_warped'] = refine_warp_network.RefineWarp.run_fast(self, example)

        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example['middle']['before_refined_warped'],
            example['middle']['after_refined_warped'],
            example['middle']['fusion']
        )
        return average, attention

    def run_attention_averaging(self, example):
        refine_warp_network.RefineWarp.run_and_pack_to_example(self, example)
        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example["middle"]["before_refined_warped"],
            example["middle"]["after_refined_warped"],
            example["middle"]["fusion"],
        )
        return average, attention

    def run_and_pack_to_example(self, example):
        (
            example["middle"]["attention_average"],
            example["middle"]["attention"],
        ) = self.run_attention_averaging(example)

    def forward(self, example):
        return self.run_attention_averaging(example)
