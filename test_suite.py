import os
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from preprocess import preprocess_signature
from model import SiameseEmbeddingNet, ContrastiveLoss
from dataset import SignaturePairsDataset
from generate_samples import create_sample_signatures

class TestOfflineSignatureSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure sample dataset exists
        cls.raw_dataset_dir = "dataset/raw"
        create_sample_signatures(cls.raw_dataset_dir)

    def test_01_preprocessing_pipeline(self):
        sample_path = os.path.join(self.raw_dataset_dir, "student_001", "g_01.png")
        self.assertTrue(os.path.exists(sample_path), "Sample signature file should exist.")

        # Test from file path
        proc = preprocess_signature(sample_path, target_size=(128, 128))
        self.assertEqual(proc.shape, (128, 128), "Preprocessed image must have shape (128, 128)")
        self.assertEqual(proc.dtype, np.float32, "Preprocessed image must be float32")
        self.assertTrue(0.0 <= proc.min() and proc.max() <= 1.0, "Pixels must be normalized within [0.0, 1.0]")
        self.assertTrue(proc.sum() > 0, "Signatures strokes should have non-zero positive pixels")

        # Test from PIL Image
        pil_img = Image.open(sample_path)
        proc_pil = preprocess_signature(pil_img, target_size=(128, 128))
        self.assertEqual(proc_pil.shape, (128, 128))
        self.assertTrue(np.allclose(proc, proc_pil, atol=1e-3), "File path and PIL preprocessing should yield identical results")

    def test_02_dataset_pair_generation(self):
        num_pairs = 40
        ds = SignaturePairsDataset(self.raw_dataset_dir, num_pairs=num_pairs)
        self.assertEqual(len(ds), num_pairs, f"Dataset should generate requested {num_pairs} pairs")

        labels = []
        for i in range(num_pairs):
            t1, t2, lbl = ds[i]
            self.assertEqual(t1.shape, (1, 128, 128), "Image 1 must have shape (1, 128, 128)")
            self.assertEqual(t2.shape, (1, 128, 128), "Image 2 must have shape (1, 128, 128)")
            self.assertIn(lbl.item(), [0.0, 1.0], "Label must be either 0.0 (genuine) or 1.0 (forgery/impostor)")
            labels.append(lbl.item())

        genuine_count = labels.count(0.0)
        impostor_count = labels.count(1.0)
        self.assertTrue(genuine_count > 0 and impostor_count > 0, "Dataset must contain both positive and negative pairs")

    def test_03_siamese_architecture(self):
        embedding_dim = 128
        model = SiameseEmbeddingNet(embedding_dim=embedding_dim)
        model.eval()

        # Check single forward pass
        dummy_x = torch.randn(2, 1, 128, 128)
        emb = model.forward_once(dummy_x)
        self.assertEqual(emb.shape, (2, embedding_dim), "Embedding output shape should be (batch_size, 128)")

        # Verify unit L2 norm
        norms = torch.norm(emb, p=2, dim=1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=4, msg="Output embeddings must be L2-normalized unit vectors")

        # Check twin forward pass
        dummy_x2 = torch.randn(2, 1, 128, 128)
        out1, out2 = model(dummy_x, dummy_x2)
        self.assertEqual(out1.shape, (2, embedding_dim))
        self.assertEqual(out2.shape, (2, embedding_dim))

    def test_04_contrastive_loss(self):
        loss_fn = ContrastiveLoss(margin=1.0)
        
        # Test 1: Identical embeddings with genuine label (0) -> Loss should be 0
        emb1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        emb2 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        label_gen = torch.tensor([0.0], dtype=torch.float32)
        loss_gen = loss_fn(emb1, emb2, label_gen)
        self.assertAlmostEqual(loss_gen.item(), 0.0, places=5, msg="Identical genuine pairs should produce zero loss")

        # Test 2: Orthogonal embeddings with forged label (1)
        # Distance = sqrt(1^2 + 1^2) = 1.414. With margin 1.0, clamp(1.0 - 1.414, min=0) = 0 -> Loss = 0
        emb_ortho = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        label_imp = torch.tensor([1.0], dtype=torch.float32)
        loss_imp = loss_fn(emb1, emb_ortho, label_imp)
        self.assertAlmostEqual(loss_imp.item(), 0.0, places=5)

    def test_05_end_to_end_verification_pipeline(self):
        # Simulate realistic end-to-end verification
        s1_g1 = os.path.join(self.raw_dataset_dir, "student_001", "g_01.png")
        s1_g2 = os.path.join(self.raw_dataset_dir, "student_001", "g_02.png")
        s1_f1 = os.path.join(self.raw_dataset_dir, "student_001", "forged", "f_01.png")

        p_g1 = torch.from_numpy(preprocess_signature(s1_g1)).unsqueeze(0).unsqueeze(0).float()
        p_g2 = torch.from_numpy(preprocess_signature(s1_g2)).unsqueeze(0).unsqueeze(0).float()
        p_f1 = torch.from_numpy(preprocess_signature(s1_f1)).unsqueeze(0).unsqueeze(0).float()

        model = SiameseEmbeddingNet(embedding_dim=128)
        model.eval()

        with torch.no_grad():
            emb_g1 = model.forward_once(p_g1)
            emb_g2 = model.forward_once(p_g2)
            emb_f1 = model.forward_once(p_f1)

            dist_genuine = F.pairwise_distance(emb_g1, emb_g2).item()
            dist_forged = F.pairwise_distance(emb_g1, emb_f1).item()

            self.assertIsInstance(dist_genuine, float)
            self.assertIsInstance(dist_forged, float)
            self.assertTrue(dist_genuine >= 0.0 and dist_forged >= 0.0)

if __name__ == "__main__":
    unittest.main()
