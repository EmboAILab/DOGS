import unittest
from pathlib import Path

import torch

from Training.dogs_identity import (
    dataset_object_ids,
    gaussian_identity_probabilities,
    hard_identity_assignment,
    hard_object_mask,
    save_gaussian_subset,
    visible_object_ids,
)


class DogsIdentityTests(unittest.TestCase):
    def test_identity_probabilities_are_per_gaussian_softmax(self) -> None:
        features = torch.tensor(
            [
                [[2.0, 0.0]],
                [[0.0, 2.0]],
                [[1.0, 1.0]],
            ]
        )
        classifier = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]]],
                        [[[0.0]], [[1.0]]],
                        [[[-1.0]], [[-1.0]]],
                    ]
                )
            )
        probabilities = gaussian_identity_probabilities(classifier, features)
        self.assertEqual(tuple(probabilities.shape), (3, 3))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(3)))
        self.assertEqual(probabilities.argmax(dim=1).tolist(), [0, 1, 0])

    def test_hard_extraction_uses_configured_probability_threshold(self) -> None:
        probabilities = torch.tensor([[0.7, 0.2], [0.3, 0.6], [0.1, 0.9]])
        self.assertEqual(hard_object_mask(probabilities, 1, 0.3).tolist(), [False, True, True])

    def test_hard_groups_are_mutually_exclusive(self) -> None:
        probabilities = torch.tensor(
            [
                [0.45, 0.40, 0.15],
                [0.20, 0.55, 0.25],
                [0.34, 0.33, 0.33],
                [0.30, 0.30, 0.40],
            ]
        )
        assignment = hard_identity_assignment(probabilities, threshold=0.4)
        self.assertEqual(assignment.tolist(), [0, 1, -1, -1])
        masks = torch.stack(
            [hard_object_mask(probabilities, object_id, 0.4) for object_id in range(3)],
            dim=1,
        )
        self.assertTrue(torch.all(masks.sum(dim=1) <= 1))

    def test_dataset_object_ids_uses_all_training_views(self) -> None:
        labels = [torch.tensor([[0, 2], [-1, 2]]), torch.tensor([[0, 4], [4, 9]])]
        self.assertEqual(
            dataset_object_ids(labels, background_label=0, num_classes=8),
            [2, 4],
        )

    def test_visible_objects_excludes_background_and_invalid_labels(self) -> None:
        labels = torch.tensor([[0, 2, 2], [-1, 4, 9]])
        self.assertEqual(visible_object_ids(labels, background_label=0, num_classes=8), [2, 4])

    def test_subset_export_restores_grouped_model(self) -> None:
        class FakeGaussians:
            def __init__(self) -> None:
                self._xyz = torch.nn.Parameter(torch.arange(12.0).reshape(4, 3))
                self._features_dc = torch.nn.Parameter(torch.zeros(4, 1, 3))
                self._features_rest = torch.nn.Parameter(torch.zeros(4, 2, 3))
                self._opacity = torch.nn.Parameter(torch.zeros(4, 1))
                self._scaling = torch.nn.Parameter(torch.zeros(4, 3))
                self._rotation = torch.nn.Parameter(torch.zeros(4, 4))
                self._objects_dc = torch.nn.Parameter(torch.zeros(4, 1, 2))
                self.max_radii2D = torch.arange(4.0)
                self.saved_count = 0

            @property
            def get_xyz(self):
                return self._xyz

            def save_ply(self, path: str) -> None:
                self.saved_count = self._xyz.shape[0]
                self.saved_path = path

        gaussians = FakeGaussians()
        original_xyz = gaussians._xyz
        count = save_gaussian_subset(
            gaussians,
            torch.tensor([True, False, True, False]),
            Path("object.ply"),
        )
        self.assertEqual(count, 2)
        self.assertEqual(gaussians.saved_count, 2)
        self.assertIs(gaussians._xyz, original_xyz)


if __name__ == "__main__":
    unittest.main()
