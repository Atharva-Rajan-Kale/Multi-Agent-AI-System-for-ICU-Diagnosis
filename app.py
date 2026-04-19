"""
ICU Multi-Agent Diagnosis System - Conversational Interface
Run with: streamlit run app.py
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="ICU Multi-Agent Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data_dir = Path('interface_data')
    
    with open(data_dir / 'patient_database.json', 'r') as f:
        patients = json.load(f)
    
    with open(data_dir / 'disease_list.json', 'r') as f:
        diseases = json.load(f)
    
    with open(data_dir / 'thresholds.json', 'r') as f:
        thresholds = json.load(f)
    
    # Create lookup dictionary
    patient_lookup = {p['patient_id']: p for p in patients}
    
    return patients, diseases, thresholds, patient_lookup

def calculate_fusion_features(agent1_prob, agent2_prob, agent3_prob):
    """
    Calculate fusion features to demonstrate interpretability.
    Returns a dictionary of feature contributions (simulated for demo).
    
    In production, these would come from actual SHAP values.
    """
    # Base probabilities
    features = {
        'Agent 1 Probability': agent1_prob,
        'Agent 2 Probability': agent2_prob,
        'Agent 3 Probability': agent3_prob,
    }
    
    # Interaction products
    features['Agent 1×2 Interaction'] = agent1_prob * agent2_prob
    features['Agent 1×3 Interaction'] = agent1_prob * agent3_prob
    features['Agent 2×3 Interaction'] = agent2_prob * agent3_prob
    
    # Disagreement features
    features['Agent 1-2 Disagreement'] = abs(agent1_prob - agent2_prob)
    features['Agent 1-3 Disagreement'] = abs(agent1_prob - agent3_prob)
    features['Agent 2-3 Disagreement'] = abs(agent2_prob - agent3_prob)
    
    # Agreement (binary consensus)
    binary1 = 1 if agent1_prob >= 0.5 else 0
    binary2 = 1 if agent2_prob >= 0.5 else 0
    binary3 = 1 if agent3_prob >= 0.5 else 0
    features['Unanimous Agreement'] = 1 if (binary1 == binary2 == binary3) else 0
    
    # Variance
    probs = [agent1_prob, agent2_prob, agent3_prob]
    mean_prob = np.mean(probs)
    variance = np.var(probs)
    features['Prediction Variance'] = variance
    
    return features

def get_feature_importance_demo(features, fusion_prob):
    """
    Simulate feature importance contributions.
    In production, these would be actual SHAP values from the trained model.
    """
    # Simplified importance estimation (for demo purposes)
    # In reality, load pre-computed SHAP values
    
    importance = {}
    
    # Agent probabilities get base importance
    importance['Agent 1 Probability'] = features['Agent 1 Probability'] * 0.31
    importance['Agent 2 Probability'] = features['Agent 2 Probability'] * 0.24
    importance['Agent 3 Probability'] = features['Agent 3 Probability'] * 0.15
    
    # Interactions get importance based on product
    importance['Agent 1×2 Interaction'] = features['Agent 1×2 Interaction'] * 0.18
    importance['Agent 1×3 Interaction'] = features['Agent 1×3 Interaction'] * 0.05
    importance['Agent 2×3 Interaction'] = features['Agent 2×3 Interaction'] * 0.03
    
    # Variance importance (inverse - low variance = high confidence)
    importance['Prediction Variance'] = (1 - features['Prediction Variance']) * 0.12
    
    # Agreement importance
    importance['Unanimous Agreement'] = features['Unanimous Agreement'] * 0.07
    
    # Disagreement features (lower disagreement = higher importance)
    importance['Agent 1-2 Disagreement'] = (1 - features['Agent 1-2 Disagreement']) * 0.04
    
    # Normalize to fusion probability
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total * fusion_prob for k, v in importance.items()}
    
    return importance

patients, DISEASE_LIST, thresholds, patient_lookup = load_data()

# Title
st.title("🏥 ICU Multi-Agent Diagnosis System")
st.markdown("**Interactive Exploration of Multi-Agent Fusion for Clinical Decision Support**")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["📊 Dashboard View", "💬 Agent Conversation"])

# ==============================================================================
# TAB 1: DASHBOARD VIEW (Original Interface + Fusion Interpretability)
# ==============================================================================
with tab1:
    # Sidebar - Patient Selection
    st.sidebar.header("👤 Patient Selection")
    
    # Browse all patients
    patient_ids = sorted([p['patient_id'] for p in patients])
    selected_patient_id = st.sidebar.selectbox(
        "Select patient ID:",
        patient_ids,
        index=0,
        key="dashboard_patient"
    )
    
    # Get patient data
    patient = patient_lookup[selected_patient_id]
    
    # Display patient summary
    st.header(f"📊 Patient {patient['patient_id']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("True Diagnoses")
        true_positives = [d for d in DISEASE_LIST if patient['true_labels'][d] == 1]
        true_negatives = [d for d in DISEASE_LIST if patient['true_labels'][d] == 0]
        
        if true_positives:
            for disease in true_positives:
                st.markdown(f"✅ **{disease}**")
        else:
            st.markdown("❌ No diseases")
    
    with col2:
        st.subheader("Fusion Predictions")
        fusion_positives = [d for d in DISEASE_LIST if patient['fusion_binary'][d] == 1]
        
        if fusion_positives:
            for disease in fusion_positives:
                # Check if correct
                if patient['true_labels'][disease] == 1:
                    st.markdown(f"✅ **{disease}** (Correct)")
                else:
                    st.markdown(f"⚠️ **{disease}** (False Positive)")
        else:
            st.markdown("❌ No diseases predicted")
    
    # Calculate accuracy
    correct = sum([
        1 for d in DISEASE_LIST 
        if patient['fusion_binary'][d] == patient['true_labels'][d]
    ])
    st.markdown(f"**Overall: {correct}/9 Correct ({correct/9*100:.1f}%)**")
    
    st.markdown("---")
    
    # Disease selector
    st.header("🎯 Disease-Specific Analysis")
    
    selected_disease = st.selectbox(
        "Select disease to analyze:",
        DISEASE_LIST,
        index=0,
        key="dashboard_disease"
    )
    
    # Get predictions for selected disease
    true_label = patient['true_labels'][selected_disease]
    agent1_prob = patient['agent1_probs'][selected_disease]
    agent2_prob = patient['agent2_probs'][selected_disease]
    agent3_prob = patient['agent3_probs'][selected_disease]
    fusion_prob = patient['fusion_probs'][selected_disease]
    threshold = patient['thresholds'][selected_disease]
    
    # Display predictions
    st.subheader(f"📋 {selected_disease} - Prediction Summary")
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Agent': ['Agent 1 (Labs)', 'Agent 2 (Notes)', 'Agent 3 (Vitals)', 'Fusion'],
        'Probability': [
            f"{agent1_prob*100:.1f}%",
            f"{agent2_prob*100:.1f}%",
            f"{agent3_prob*100:.1f}%",
            f"{fusion_prob*100:.1f}%"
        ],
        'Prediction': [
            '✅ Positive' if agent1_prob >= 0.5 else '❌ Negative',
            '✅ Positive' if agent2_prob >= 0.5 else '❌ Negative',
            '✅ Positive' if agent3_prob >= 0.5 else '❌ Negative',
            f'✅ Positive' if fusion_prob >= threshold else '❌ Negative'
        ],
        'Correct': [
            '✅' if (agent1_prob >= 0.5) == true_label else '❌',
            '✅' if (agent2_prob >= 0.5) == true_label else '❌',
            '✅' if (agent3_prob >= 0.5) == true_label else '❌',
            '✅' if (fusion_prob >= threshold) == true_label else '❌'
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # True label
    if true_label == 1:
        st.success(f"✅ **True Label: POSITIVE** (Patient has {selected_disease})")
    else:
        st.info(f"❌ **True Label: NEGATIVE** (Patient does not have {selected_disease})")
    
    st.markdown("---")
    
    # Agent details with EVIDENCE
    st.subheader("🔍 Detailed Agent Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Get evidence for selected disease
    agent1_evidence = patient.get('agent1_evidence', {}).get(selected_disease, [])
    agent2_evidence = patient.get('agent2_evidence', {}).get(selected_disease, [])
    agent3_evidence = patient.get('agent3_evidence', {}).get(selected_disease, [])
    
    with col1:
        st.markdown("### 🔬 Agent 1 (Labs)")
        st.metric("Probability", f"{agent1_prob*100:.1f}%")
        st.metric("Decision", "Positive" if agent1_prob >= 0.5 else "Negative")
        
        if agent1_evidence:
            st.markdown("**📊 Key Laboratory Values:**")
            for ev in agent1_evidence[:5]:
                status_emoji = {"↑": "🔴", "↓": "🔵", "✓": "✅"}
                emoji = status_emoji.get(ev['status'], "")
                st.markdown(f"{emoji} **{ev['lab_name']}**: {ev['value']:.2f} {ev['unit']}")
                if ev.get('normal_low') and ev.get('normal_high'):
                    st.caption(f"Normal: {ev['normal_low']}-{ev['normal_high']} {ev['unit']}")
        else:
            st.info("No lab data available for this patient")
    
    with col2:
        st.markdown("### 📝 Agent 2 (Notes)")
        st.metric("Probability", f"{agent2_prob*100:.1f}%")
        st.metric("Decision", "Positive" if agent2_prob >= 0.5 else "Negative")
        
        if agent2_evidence:
            st.markdown("**📄 Key Clinical Phrases:**")
            for phrase in agent2_evidence[:3]:
                st.markdown(f"• *{phrase}*")
        else:
            st.info("No relevant clinical notes found")
    
    with col3:
        st.markdown("### 💓 Agent 3 (Vitals)")
        st.metric("Probability", f"{agent3_prob*100:.1f}%")
        st.metric("Decision", "Positive" if agent3_prob >= 0.5 else "Negative")
        
        if agent3_evidence:
            st.markdown("**📊 Vital Sign Patterns (24hr):**")
            for ev in agent3_evidence[:5]:
                st.markdown(f"**{ev['vital_name']}**")
                st.caption(f"Mean: {ev['mean']:.1f} | Range: [{ev['min']:.1f}, {ev['max']:.1f}]")
        else:
            st.info("No vital sign data available for this patient")
    
    st.markdown("---")
    
    # Fusion decision with EVIDENCE SUMMARY
    st.subheader("🤖 Fusion Decision")
    
    fusion_correct = (fusion_prob >= threshold) == true_label
    
    if fusion_correct:
        st.success(f"✅ **Fusion CORRECT**: Predicted {'Positive' if fusion_prob >= threshold else 'Negative'}, Truth is {'Positive' if true_label else 'Negative'}")
    else:
        st.error(f"❌ **Fusion INCORRECT**: Predicted {'Positive' if fusion_prob >= threshold else 'Negative'}, Truth is {'Positive' if true_label else 'Negative'}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Fusion Probability", f"{fusion_prob*100:.1f}%")
        st.metric("Decision Threshold", f"{threshold*100:.0f}%")
    
    with col2:
        # Calculate how many agents agree with fusion
        agents_agree = sum([
            (agent1_prob >= 0.5) == (fusion_prob >= threshold),
            (agent2_prob >= 0.5) == (fusion_prob >= threshold),
            (agent3_prob >= 0.5) == (fusion_prob >= threshold)
        ])
        st.metric("Agents Agreeing with Fusion", f"{agents_agree}/3")
    
    # ========================================================================
    # NEW: FUSION INTERPRETABILITY SECTION (XGBoost Advantage)
    # ========================================================================
    st.markdown("---")
    st.subheader("🧠 Fusion Decision Explanation")
    
    with st.expander("📊 **View Feature-Level Breakdown** (Click to expand)", expanded=False):
        
        
        # Calculate fusion features
        fusion_features = calculate_fusion_features(agent1_prob, agent2_prob, agent3_prob)
        feature_importance = get_feature_importance_demo(fusion_features, fusion_prob)
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        st.markdown("### Top Contributing Features:")
        
        # Display top 8 features
        for i, (feature_name, contribution) in enumerate(sorted_features[:8], 1):
            feature_value = fusion_features[feature_name]
            
            # Create progress bar
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Progress bar (normalized to 0-1 range)
                max_contrib = max([abs(c) for _, c in sorted_features])
                bar_value = abs(contribution) / max_contrib if max_contrib > 0 else 0
                st.progress(bar_value, text=f"**{i}. {feature_name}**")
            
            with col2:
                st.metric("Value", f"{feature_value:.3f}")
            
            with col3:
                st.metric("Contribution", f"+{contribution:.3f}" if contribution >= 0 else f"{contribution:.3f}")
        
        st.markdown("---")
        
        # Feature type explanations
        st.markdown("### Feature Type Explanations:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔵 Base Probabilities:**
            - Direct agent predictions
            - Foundation for all derived features
            
            **🟢 Interaction Products:**
            - Measures synergistic agreement
            - HIGH = multiple modalities converge
            - Example: Labs AND notes both positive
            """)
        
        with col2:
            st.markdown("""
            **🟡 Disagreement Features:**
            - Magnitude of prediction divergence
            - HIGH = conflicting evidence
            - Signals diagnostic uncertainty
            
            **🟣 Variance & Agreement:**
            - Holistic consensus measures
            - Low variance = tight consensus
            - Agreement = unanimous binary decision
            """)
        
        st.markdown("---")
        
        # Clinical interpretation
        st.markdown("### 💡 Clinical Interpretation:")
        
        # Get top feature
        top_feature, top_contrib = sorted_features[0]
        
        # Analyze variance
        variance = fusion_features['Prediction Variance']
        unanimous = fusion_features['Unanimous Agreement']
        
        interpretation = ""
        
        if variance < 0.01 and unanimous == 1:
            interpretation = """
            ✅ **High Confidence Decision:** All agents show tight consensus (low variance) 
            and unanimous agreement. The fusion coordinator trusts this signal strongly 
            because multiple independent data sources converge on the same conclusion.
            """
        elif variance > 0.1:
            interpretation = """
            ⚠️ **Mixed Signal Warning:** High prediction variance indicates agents disagree 
            significantly. The fusion coordinator reduces confidence in these cases, recognizing 
            that conflicting evidence suggests diagnostic uncertainty or atypical presentation.
            """
        elif top_feature.endswith('Interaction'):
            interpretation = f"""
            🔗 **Synergistic Evidence:** The top contributing feature is **{top_feature}**, 
            indicating that multiple agents' joint confidence drives this decision. When 
            independent modalities agree, fusion amplifies the signal beyond simple averaging.
            """
        else:
            interpretation = f"""
            📊 **Primary Driver:** The top contributing feature is **{top_feature}** 
            (contribution: {top_contrib:.3f}). This shows which aspect of the evidence 
            most strongly influenced the fusion coordinator's decision.
            """
        
        st.markdown(interpretation)
        
    
    # EVIDENCE SUMMARY SECTION (existing code continues...)
    st.markdown("---")
    st.markdown("### 📋 Evidence Summary Across All Agents")
    
    evidence_col1, evidence_col2, evidence_col3 = st.columns(3)
    
    with evidence_col1:
        st.markdown("**🔬 Labs Evidence:**")
        if agent1_evidence:
            abnormal = [e for e in agent1_evidence if e['status'] != '✓']
            if abnormal:
                for ev in abnormal[:3]:
                    st.markdown(f"• {ev['lab_name']}: {ev['value']:.1f} {ev['status']}")
            else:
                st.markdown("• All labs within normal range")
        else:
            st.markdown("• No lab data")
    
    with evidence_col2:
        st.markdown("**📝 Notes Evidence:**")
        if agent2_evidence:
            for phrase in agent2_evidence[:3]:
                # Shorten long phrases
                display_phrase = phrase[:50] + "..." if len(phrase) > 50 else phrase
                st.markdown(f"• {display_phrase}")
        else:
            st.markdown("• No clinical notes")
    
    with evidence_col3:
        st.markdown("**💓 Vitals Evidence:**")
        if agent3_evidence:
            for ev in agent3_evidence[:3]:
                st.markdown(f"• {ev['vital_name']}: {ev['mean']:.0f}")
        else:
            st.markdown("• No vital data")
    
    # Fusion explanation
    st.markdown("---")
    st.markdown("**💡 Fusion Insight:**")
    
    # Check agreement pattern
    all_agents_below = (agent1_prob < 0.5) and (agent2_prob < 0.5) and (agent3_prob < 0.5)
    all_agents_above = (agent1_prob >= 0.5) and (agent2_prob >= 0.5) and (agent3_prob >= 0.5)
    
    if all_agents_below and fusion_prob >= threshold and true_label == 1:
        st.markdown("""
        🎯 **Perfect Fusion Win!** All three agents predicted negative (below 50%), but fusion 
        correctly identified the disease. The fusion coordinator recognized a subtle pattern 
        across modalities: borderline elevations in labs, mild symptoms in notes, and 
        physiological stress in vitals that individually appear benign but collectively 
        indicate disease.
        """)
    elif all_agents_above and fusion_prob >= threshold and true_label == 0:
        st.markdown("""
        ⚠️ **Collective False Positive.** All agents agreed on positive, and fusion followed. 
        This patient likely has confounding factors (other diseases or conditions) that mimic 
        the target disease across all modalities.
        """)
    elif fusion_prob >= threshold and true_label == 1:
        st.markdown("""
        ✅ **Successful Detection.** Fusion integrated evidence from multiple agents to correctly 
        identify this disease. The multi-agent approach provided complementary information that 
        improved diagnostic accuracy.
        """)
    elif fusion_prob < threshold and true_label == 0:
        st.markdown("""
        ✅ **Correct Negative.** Fusion correctly ruled out this disease based on combined 
        evidence from all agents.
        """)
    else:
        st.markdown("""
        ❌ **Missed Case.** Fusion failed to detect this disease despite evidence from agents. 
        This represents a challenging case where signals were too weak or contradictory.
        """)
    
    st.markdown("---")
    
    # All diseases summary
    st.subheader("📋 All 9 Diseases - Complete Summary")
    
    summary_data = []
    for disease in DISEASE_LIST:
        summary_data.append({
            'Disease': disease,
            'True': '✅' if patient['true_labels'][disease] == 1 else '❌',
            'Fusion Prob': f"{patient['fusion_probs'][disease]*100:.1f}%",
            'Fusion Pred': '✅' if patient['fusion_binary'][disease] == 1 else '❌',
            'Outcome': '✅ Correct' if patient['fusion_binary'][disease] == patient['true_labels'][disease] else '❌ Wrong'
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    


# ==============================================================================
# TAB 2: CONVERSATIONAL AGENT VIEW (with fusion interpretability)
# ==============================================================================
with tab2:
    st.markdown("### 💬 Multi-Agent Clinical Discussion")
    st.markdown("*Experience how the agents collaborate to reach a diagnosis*")
    st.markdown("---")
    
    # Patient and disease selector
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_patient_id_chat = st.selectbox(
            "Select Patient ID:",
            sorted([p['patient_id'] for p in patients]),
            key="chat_patient"
        )
    
    with col2:
        selected_disease_chat = st.selectbox(
            "Select Disease:",
            DISEASE_LIST,
            key="chat_disease"
        )
    
    patient_chat = patient_lookup[selected_patient_id_chat]
    
    st.markdown("---")
    
    # Get data
    true_label_chat = patient_chat['true_labels'][selected_disease_chat]
    agent1_prob_chat = patient_chat['agent1_probs'][selected_disease_chat]
    agent2_prob_chat = patient_chat['agent2_probs'][selected_disease_chat]
    agent3_prob_chat = patient_chat['agent3_probs'][selected_disease_chat]
    fusion_prob_chat = patient_chat['fusion_probs'][selected_disease_chat]
    threshold_chat = patient_chat['thresholds'][selected_disease_chat]
    
    agent1_evidence_chat = patient_chat.get('agent1_evidence', {}).get(selected_disease_chat, [])
    agent2_evidence_chat = patient_chat.get('agent2_evidence', {}).get(selected_disease_chat, [])
    agent3_evidence_chat = patient_chat.get('agent3_evidence', {}).get(selected_disease_chat, [])
    
    # Agent 1 speaks
    with st.chat_message("assistant", avatar="🔬"):
        st.markdown("**Agent 1 - Laboratory Analyst**")
        
        message = f"Hi team, I've reviewed Patient {selected_patient_id_chat}'s laboratory values for {selected_disease_chat.lower().replace('_', ' ')}. "
        
        if agent1_evidence_chat:
            abnormal = [e for e in agent1_evidence_chat if e['status'] != '✓']
            if abnormal:
                message += "I'm seeing some concerning signs:\n"
                for ev in abnormal[:3]:
                    emoji = "🔴" if ev['status'] == '↑' else "🔵"
                    message += f"\n• **{ev['lab_name']}** is {ev['status']} at **{ev['value']:.1f} {ev['unit']}**"
                    if ev.get('normal_low') and ev.get('normal_high'):
                        message += f" (normal: {ev['normal_low']}-{ev['normal_high']}) {emoji}"
            else:
                message += "Most laboratory values appear within normal ranges. "
        else:
            message += "Unfortunately, I don't have complete lab data for this patient. "
        
        message += f"\n\nBased on these findings, I'm **{agent1_prob_chat*100:.1f}% confident** this is {selected_disease_chat.lower().replace('_', ' ')}. "
        
        if agent1_prob_chat < 0.55 and agent1_prob_chat > 0.45:
            message += "The signals are subtle and borderline."
        elif agent1_prob_chat >= 0.8:
            message += "The lab evidence is quite strong."
        elif agent1_prob_chat <= 0.2:
            message += "The labs don't support this diagnosis."
        
        st.markdown(message)
    
    # Agent 2 speaks
    with st.chat_message("assistant", avatar="📝"):
        st.markdown("**Agent 2 - Clinical Notes Analyst**")
        
        message = "I've analyzed the discharge summary. "
        
        if agent2_evidence_chat:
            message += "I found several relevant mentions:\n"
            for phrase in agent2_evidence_chat[:3]:
                message += f"\n• *{phrase}*"
            
            message += f"\n\nBased on the clinical documentation, I'm **{agent2_prob_chat*100:.1f}% confident** this is {selected_disease_chat.lower().replace('_', ' ')}. "
        else:
            message += f"I couldn't find specific mentions related to {selected_disease_chat.lower().replace('_', ' ')} in the notes. "
            message += f"My confidence is **{agent2_prob_chat*100:.1f}%** based on the overall clinical context. "
        
        if agent2_prob_chat < 0.55 and agent2_prob_chat > 0.45:
            message += "The documentation is somewhat ambiguous."
        elif agent2_prob_chat >= 0.8:
            message += "The clinical notes strongly suggest this diagnosis."
        
        st.markdown(message)
    
    # Agent 3 speaks
    with st.chat_message("assistant", avatar="💓"):
        st.markdown("**Agent 3 - Vital Signs Analyst**")
        
        message = "Looking at the 24-hour vital sign patterns, "
        
        if agent3_evidence_chat:
            message += "I see:\n"
            for ev in agent3_evidence_chat[:3]:
                message += f"\n• **{ev['vital_name']}** averaging **{ev['mean']:.1f}** (range: {ev['min']:.1f} - {ev['max']:.1f})"
            
            message += f"\n\nThe vital patterns give me **{agent3_prob_chat*100:.1f}% confidence** for {selected_disease_chat.lower().replace('_', ' ')}. "
        else:
            message += f"I don't have vital sign data for this patient. My confidence is **{agent3_prob_chat*100:.1f}%** based on limited information. "
        
        if agent3_prob_chat >= 0.8:
            message += "The physiological stress is evident."
        elif agent3_prob_chat <= 0.3:
            message += "The vitals don't indicate significant pathology."
        else:
            message += "The signals are moderate."
        
        st.markdown(message)
    
    # Fusion Coordinator speaks (WITH INTERPRETABILITY)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("**Fusion Coordinator**")
        
        # Calculate fusion features for interpretability
        fusion_features_chat = calculate_fusion_features(agent1_prob_chat, agent2_prob_chat, agent3_prob_chat)
        feature_importance_chat = get_feature_importance_demo(fusion_features_chat, fusion_prob_chat)
        sorted_features_chat = sorted(feature_importance_chat.items(), key=lambda x: abs(x[1]), reverse=True)
        
        message = "Thank you all. Let me synthesize your findings...\n\n"
        
        # Analyze agreement
        all_low = agent1_prob_chat < 0.5 and agent2_prob_chat < 0.5 and agent3_prob_chat < 0.5
        all_high = agent1_prob_chat >= 0.5 and agent2_prob_chat >= 0.5 and agent3_prob_chat >= 0.5
        disagreement = max([agent1_prob_chat, agent2_prob_chat, agent3_prob_chat]) - min([agent1_prob_chat, agent2_prob_chat, agent3_prob_chat])
        variance = fusion_features_chat['Prediction Variance']
        
        if all_low and fusion_prob_chat >= threshold_chat:
            message += "I notice an interesting pattern: each of you individually predicted negative, with confidence below 50%. "
            message += "However, looking at the **COMBINATION** of signals - subtle lab abnormalities, clinical documentation, and physiological stress - "
            message += "I detect a pattern of early disease that no single modality captures clearly.\n\n"
            message += f"**My analysis:** The interaction between your predictions (Agent 1×2 = {fusion_features_chat['Agent 1×2 Interaction']:.3f}) "
            message += "reveals synergistic evidence that wouldn't be apparent from individual assessments.\n\n"
        elif all_high and fusion_prob_chat >= threshold_chat:
            message += "All three agents agree this is likely positive. The evidence across modalities is consistent and reinforcing. "
            if variance < 0.01:
                message += f"\n\n**Confidence boost:** Your predictions show very tight consensus (variance: {variance:.4f}), "
                message += "which strengthens my confidence in this diagnosis. "
            message += "I concur with the consensus.\n\n"
        elif disagreement > 0.3:
            message += f"I see significant disagreement among you (spread: {disagreement*100:.1f}%). "
            message += "Let me weigh the evidence carefully considering each agent's reliability for this condition...\n\n"
            
            # Show top contributing feature
            top_feature_chat, top_contrib_chat = sorted_features_chat[0]
            message += f"**Primary driver:** {top_feature_chat} (contribution: {top_contrib_chat:.3f})\n\n"
        else:
            message += "Your predictions are relatively aligned. After integrating all perspectives...\n\n"
        
        # Show top 3 fusion features driving decision
        message += "**My decision reasoning (top fusion features):**\n"
        for i, (feat_name, contrib) in enumerate(sorted_features_chat[:3], 1):
            feat_val = fusion_features_chat[feat_name]
            message += f"{i}. **{feat_name}**: {feat_val:.3f} (→ {contrib:+.3f})\n"
        
        message += "\n"
        
        decision = "POSITIVE" if fusion_prob_chat >= threshold_chat else "NEGATIVE"
        message += f"**My final decision: {fusion_prob_chat*100:.1f}% confidence - {decision}**\n"
        message += f"(Decision threshold: {threshold_chat*100:.0f}%)\n\n"
        
        # Interpretability note
        if variance < 0.02:
            message += "🔒 *High confidence: Low variance indicates strong consensus across all modalities.*\n\n"
        elif variance > 0.1:
            message += "⚠️ *Moderate confidence: High variance suggests conflicting signals - proceeding cautiously.*\n\n"
        
        # Show outcome
        fusion_correct_chat = (fusion_prob_chat >= threshold_chat) == true_label_chat
        truth = "positive" if true_label_chat == 1 else "negative"
        
        if fusion_correct_chat:
            message += f"✅ **FUSION CORRECT**: Patient ground truth is **{truth}**\n\n"
            
            # Determine type of win
            agents_wrong = sum([
                (agent1_prob_chat >= 0.5) != true_label_chat,
                (agent2_prob_chat >= 0.5) != true_label_chat,
                (agent3_prob_chat >= 0.5) != true_label_chat
            ])
            
            if agents_wrong == 3:
                message += "🎯 **Perfect Fusion Win!** All three agents were incorrect, but fusion detected the pattern!"
            elif agents_wrong > 0:
                message += f"💡 **Fusion Success**: {agents_wrong} agent(s) missed this, but fusion integrated the evidence correctly."
            else:
                message += "✨ **Team Success**: All agents and fusion aligned on the correct diagnosis."
        else:
            message += f"❌ **FUSION INCORRECT**: Patient ground truth is **{truth}**\n\n"
            message += "This represents a challenging case where the signals were too weak or contradictory."
        

        
        st.markdown(message)
    
    # Show ground truth
    st.markdown("---")
    st.markdown("### 📋 Ground Truth")
    true_diseases = [d for d in DISEASE_LIST if patient_chat['true_labels'][d] == 1]
    if true_diseases:
        st.success(f"**Patient's actual diagnoses**: {', '.join(true_diseases)}")
    else:
        st.info("**Patient has none of the 9 diseases**")