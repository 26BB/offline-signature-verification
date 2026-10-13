import os
import glob
import random
import torch
from torch.utils.data import Dataset
from preprocess import preprocess_signature

class SignaturePairsDataset(Dataset):
    """
    Pairs dataset generator for Siamese Network training.
    Assumes directory format:
    dataset_dir/
      student_001/
        g_01.png ... g_20.png  (genuine signatures)
        forged/
          f_01.png ... f_05.png (forged signatures)
    """
    def __init__(self, root_dir, num_pairs=2000, is_training=True):
        self.root_dir = root_dir
        self.num_pairs = num_pairs
        self.is_training = is_training
        
        # Discover students
        self.students = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        
        self.genuine_dict = {}
        self.forged_dict = {}

        for s in self.students:
            s_path = os.path.join(root_dir, s)
            # Find genuine images in student directory (excluding forged subfolder)
            gen_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                gen_files.extend(glob.glob(os.path.join(s_path, ext)))
            self.genuine_dict[s] = gen_files

            # Find forged images in forged subfolder
            forged_files = []
            forged_dir = os.path.join(s_path, "forged")
            if os.path.isdir(forged_dir):
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    forged_files.extend(glob.glob(os.path.join(forged_dir, ext)))
            self.forged_dict[s] = forged_files

        # Filter out students with zero genuine samples
        self.valid_students = [s for s in self.students if len(self.genuine_dict[s]) >= 2]
        
        # Pre-generate pair index references
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        if not self.valid_students:
            return pairs

        for _ in range(self.num_pairs):
            # 50% Genuine pairs (label 0), 50% Impostor/Forged pairs (label 1)
            is_genuine = random.random() < 0.5
            
            if is_genuine:
                # Pick same student, 2 different genuine signatures
                student = random.choice(self.valid_students)
                imgs = random.sample(self.genuine_dict[student], 2)
                pairs.append((imgs[0], imgs[1], 0.0))
            else:
                # Impostor: either forged signature of student, or signature of a different student
                student = random.choice(self.valid_students)
                img1 = random.choice(self.genuine_dict[student])
                
                # Check if forged samples exist for this student and flip coin
                if self.forged_dict.get(student) and random.random() < 0.5:
                    img2 = random.choice(self.forged_dict[student])
                else:
                    other_students = [s for s in self.valid_students if s != student]
                    if other_students:
                        other = random.choice(other_students)
                        img2 = random.choice(self.genuine_dict[other])
                    else:
                        img2 = img1  # Fallback if only 1 student
                pairs.append((img1, img2, 1.0))
                
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        
        img1 = preprocess_signature(path1)
        img2 = preprocess_signature(path2)
        
        # Add channel dimension: (1, 128, 128)
        tensor1 = torch.from_numpy(img1).unsqueeze(0).float()
        tensor2 = torch.from_numpy(img2).unsqueeze(0).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return tensor1, tensor2, label_tensor
