class SubjectEmbeddings(nn.Module):
    """
    Simple Subject Embeddings for IRT Models
    
    Converts sparse multi-subject questions into dense vector representations
    by learning embeddings for each mathematical subject.
    """
    
    def __init__(self, num_subjects: int = 288, embed_dim: int = 64):
        """
        Args:
            num_subjects: Number of unique subjects (exactly what we need)
            embed_dim: Size of embedding vectors (64 is good default)
        """
        super().__init__()
        
        self.num_subjects = num_subjects
        self.embed_dim = embed_dim
        
        # Main embedding layer - each subject gets its own vector
        self.subject_embeddings = nn.Embedding(num_subjects, embed_dim)
        
        # Initialize embeddings randomly
        nn.init.xavier_uniform_(self.subject_embeddings.weight)
    
    def forward(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert question subjects to dense embedding
        
        Args:
            subject_ids: List of subject IDs for one question
                        Example: tensor([1, 39, 98]) for "Number", "Geometry", "Basic Arithmetic"
        
        Returns:
            question_embedding: Dense vector representing the question [embed_dim]
        """
        # Get embedding vector for each subject
        subject_vectors = self.subject_embeddings(subject_ids)  # Shape: [num_subjects, embed_dim]
        
        # Average all subject vectors to get question representation
        question_embedding = torch.mean(subject_vectors, dim=0)  # Shape: [embed_dim]
        
        return question_embedding
    
    def get_batch_embeddings(self, batch_subject_lists: List[List[int]]) -> torch.Tensor:
        """
        Process multiple questions at once
        
        Args:
            batch_subject_lists: List of subject lists
                                Example: [[1, 39, 98], [17, 104], [1, 8, 242]]
        
        Returns:
            batch_embeddings: Tensor of shape [batch_size, embed_dim]
        """
        embeddings = []
        
        for subject_list in batch_subject_lists:
            subject_tensor = torch.tensor(subject_list, dtype=torch.long)
            question_emb = self.forward(subject_tensor)
            embeddings.append(question_emb)
        
        return torch.stack(embeddings)
    
    def get_similar_subjects(self, subject_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find subjects with similar embeddings
        
        Args:
            subject_id: Subject to find similarities for
            top_k: Number of similar subjects to return
        
        Returns:
            List of (subject_id, similarity_score) tuples
        """
        with torch.no_grad():
            # Get all embeddings
            all_embeddings = self.subject_embeddings.weight.numpy()
            
            # Calculate similarities
            target_embedding = all_embeddings[subject_id].reshape(1, -1)
            similarities = cosine_similarity(target_embedding, all_embeddings)[0]
            
            # Get top-k (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            results = [(idx, similarities[idx]) for idx in top_indices]
            
        return results


class SimplePretrainer:
    """
    Simple pre-training for subject embeddings
    
    Teaches the model that subjects appearing together in questions
    should have similar embeddings.
    """
    
    def __init__(self, subject_embeddings: SubjectEmbeddings, learning_rate: float = 0.001):
        self.model = subject_embeddings
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def create_training_data(self, question_subjects: List[List[int]]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Create positive and negative pairs for training
        
        Args:
            question_subjects: List of subject lists for each question
        
        Returns:
            pairs: Subject pairs
            labels: 1 for subjects that co-occur, 0 for random pairs
        """
        positive_pairs = []
        
        # Positive pairs: subjects that appear together in questions
        for subjects in question_subjects:
            for i in range(len(subjects)):
                for j in range(i + 1, len(subjects)):
                    positive_pairs.append((subjects[i], subjects[j]))
        
        # Negative pairs: random subject combinations
        all_subjects = set()
        for subjects in question_subjects:
            all_subjects.update(subjects)
        all_subjects = list(all_subjects)
        
        negative_pairs = []
        for _ in range(len(positive_pairs)):
            s1, s2 = np.random.choice(all_subjects, 2, replace=False)
            negative_pairs.append((s1, s2))
        
        # Combine
        pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        
        return pairs, labels
    
    def train(self, question_subjects: List[List[int]], epochs: int = 50):
        """
        Train the embeddings
        
        Args:
            question_subjects: List of subject lists from your data
            epochs: Number of training iterations
        """
        # Create training data
        pairs, labels = self.create_training_data(question_subjects)
        
        pairs_tensor = torch.tensor(pairs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        print(f"Training on {len(pairs)} subject pairs for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(pairs))
            pairs_shuffled = pairs_tensor[indices]
            labels_shuffled = labels_tensor[indices]
            
            # Process in batches
            batch_size = 256
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs_shuffled[i:i+batch_size]
                batch_labels = labels_shuffled[i:i+batch_size]
                
                # Get embeddings for both subjects in each pair
                emb1 = self.model.subject_embeddings(batch_pairs[:, 0])
                emb2 = self.model.subject_embeddings(batch_pairs[:, 1])
                
                # Calculate similarity
                similarity = F.cosine_similarity(emb1, emb2)
                
                # Loss: similar subjects should have high similarity, different subjects should have low similarity
                loss = F.binary_cross_entropy_with_logits(similarity, batch_labels)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(pairs) // batch_size + 1)
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        print("Training completed!")
        
if __name__ == "__main__":
    
    print("=== Subject Embeddings Demo ===")
    