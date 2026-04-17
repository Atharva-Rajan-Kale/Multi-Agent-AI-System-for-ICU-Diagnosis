"""
Agent Communication Protocol

Defines structured messages that agents use to share information
with the fusion coordinator.
"""

# Disease list (consistent order across all agents)
DISEASE_LIST = [
    'SEPSIS',
    'PNEUMONIA', 
    'RESPIRATORY_FAILURE',
    'ACUTE_KIDNEY_INJURY',
    'HEART_FAILURE',
    'ATRIAL_FIBRILLATION',
    'CORONARY_ARTERY_DISEASE',
    'ANEMIA',
    'PANCREATITIS'
]


class AgentMessage:
    """
    Structured message from an agent containing prediction and evidence
    
    Attributes:
        agent_name (str): Identifier for the agent (e.g., 'agent1_labs')
        prediction (str): Top predicted disease
        confidence (float): Confidence in top prediction (0-1)
        probabilities (dict): Probabilities for all 9 diseases
        top_features (list): Top features that influenced decision
        metadata (dict): Additional info (model type, data availability, etc.)
    """
    
    def __init__(self, agent_name, prediction, confidence, 
                 probabilities, top_features=None, metadata=None):
        self.agent_name = agent_name
        self.prediction = prediction
        self.confidence = confidence
        self.probabilities = probabilities
        self.top_features = top_features or []
        self.metadata = metadata or {}
        
        # Validate
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        if prediction not in DISEASE_LIST:
            raise ValueError(f"Prediction must be one of {DISEASE_LIST}, got {prediction}")
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'agent_name': self.agent_name,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'top_features': self.top_features,
            'metadata': self.metadata
        }
    
    def get_feature_vector(self):
        """
        Convert to feature vector for fusion ML model
        
        Returns:
            list: Feature vector containing:
                - Probabilities for all 9 diseases (9 features)
                - Confidence score (1 feature)
                - Data availability flag (1 feature)
        """
        features = []
        
        # Add probabilities in consistent order
        for disease in DISEASE_LIST:
            features.append(self.probabilities.get(disease, 0.0))
        
        # Add confidence
        features.append(self.confidence)
        
        # Add data availability flag
        features.append(1.0 if self.metadata.get('data_available', True) else 0.0)
        
        return features
    
    def __repr__(self):
        return f"AgentMessage(agent={self.agent_name}, prediction={self.prediction}, confidence={self.confidence:.3f})"


def create_fusion_features(agent1_msg, agent2_msg, agent3_msg):
    """
    Convert three agent messages into feature vector for fusion model
    
    Args:
        agent1_msg (dict): Agent 1 message dictionary
        agent2_msg (dict): Agent 2 message dictionary
        agent3_msg (dict): Agent 3 message dictionary
    
    Returns:
        list: Feature vector with 33 features:
            - Agent 1 probabilities (9)
            - Agent 2 probabilities (9)
            - Agent 3 probabilities (9)
            - Agent 1 confidence (1)
            - Agent 2 confidence (1)
            - Agent 3 confidence (1)
            - Data availability flags (3)
    """
    features = []
    
    # Add all agent probabilities
    for disease in DISEASE_LIST:
        features.append(agent1_msg['probabilities'][disease])
    
    for disease in DISEASE_LIST:
        features.append(agent2_msg['probabilities'][disease])
    
    for disease in DISEASE_LIST:
        features.append(agent3_msg['probabilities'][disease])
    
    # Add confidences
    features.append(agent1_msg['confidence'])
    features.append(agent2_msg['confidence'])
    features.append(agent3_msg['confidence'])
    
    # Add data availability flags
    features.append(1.0 if agent1_msg['metadata'].get('data_available', True) else 0.0)
    features.append(1.0 if agent2_msg['metadata'].get('data_available', True) else 0.0)
    features.append(1.0 if agent3_msg['metadata'].get('data_available', True) else 0.0)
    
    return features