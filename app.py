import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pregnancy Risk Assessment",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and artifacts
@st.cache_resource
def load_model():
    try:
        model = joblib.load('pregnancy_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Try to load feature_columns.pkl, but have a fallback
        try:
            feature_cols = joblib.load('feature_columns.pkl')
        except FileNotFoundError:
            # Fallback to default features
            feature_cols = [
                "age", "blood_pressure", "diastolic", "blood_sugar", 
                "body_temp", "bmi", "heart_rate", "previous_complications",
                "preexisting_diabetes", "gestational_diabetes", "mental_health"
            ]
            
        return model, scaler, label_encoder, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Add sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/pregnancy.png", width=100)
st.sidebar.title("About")
st.sidebar.info(
    "This application uses machine learning to assess pregnancy risk levels. "
    "Enter patient information to get a risk assessment."
)
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Enter patient details in the form\n"
    "2. Click 'Predict Risk Level'\n"
    "3. Review the risk assessment and key factors"
)

# Main content
st.title("🤰 Pregnancy Risk Assessment")
st.write("Enter patient information to predict pregnancy risk level")

try:
    model, scaler, label_encoder, feature_cols = load_model()
    
    # Create input fields for all features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=15, max_value=60, value=25)
        blood_pressure = st.number_input("Systolic Blood Pressure", min_value=70, max_value=200, value=120)
        diastolic = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
        blood_sugar = st.number_input("Blood Sugar", min_value=4.0, max_value=20.0, value=7.0, step=0.1)
    
    with col2:
        st.subheader("Physical Measurements")
        body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=104.0, value=98.6, step=0.1)
        bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=23.0, step=0.1)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    
    with col3:
        st.subheader("Medical History")
        previous_complications = st.checkbox("Previous Pregnancy Complications")
        preexisting_diabetes = st.checkbox("Preexisting Diabetes")
        gestational_diabetes = st.checkbox("Gestational Diabetes")
        mental_health = st.checkbox("Mental Health Issues")
    
    # Predict button
    if st.button("Predict Risk Level", type="primary"):
        # Create input data dictionary
        input_data = {
            'age': age,
            'blood_pressure': blood_pressure,
            'diastolic': diastolic,
            'blood_sugar': blood_sugar,
            'body_temp': body_temp,
            'bmi': bmi,
            'heart_rate': heart_rate,
            'previous_complications': int(previous_complications),
            'preexisting_diabetes': int(preexisting_diabetes),
            'gestational_diabetes': int(gestational_diabetes),
            'mental_health': int(mental_health)
        }
        
        # Create DataFrame with input data
        input_df = pd.DataFrame([input_data])
        
        # Filter to include only the features used by the model
        input_features = [col for col in feature_cols if col in input_df.columns]
        X = input_df[input_features]
        
        # Scale the input data
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Get risk level
        risk_level = label_encoder.inverse_transform([prediction])[0]
        risk_prob = probabilities[prediction]
        
        # Display result
        st.markdown("---")
        st.subheader("Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if risk_level.lower() == 'high':
                st.error(f"⚠️ **Risk Level: {risk_level.upper()}**")
                st.markdown(f"Confidence: {risk_prob:.2%}")
                st.markdown(
                    "**Recommended Action:**\n"
                    "- Immediate referral to high-risk obstetrics\n"
                    "- Schedule follow-up within 1 week\n"
                    "- Consider additional testing and monitoring"
                )
            else:
                st.success(f"✅ **Risk Level: {risk_level.upper()}**")
                st.markdown(f"Confidence: {risk_prob:.2%}")
                st.markdown(
                    "**Recommended Action:**\n"
                    "- Continue routine prenatal care\n"
                    "- Follow standard monitoring guidelines\n"
                    "- Next appointment in 4 weeks"
                )
        
        with col2:
            # Display feature importance
            feature_importance = pd.DataFrame({
                'Feature': input_features,
                'Value': X.values[0],
                'Importance (%)': model.feature_importances_ * 100
            }).sort_values('Importance (%)', ascending=False)
            
            st.subheader("Key Factors")
            st.dataframe(feature_importance.style.format({
                'Importance (%)': '{:.1f}%'
            }))
            
            # Matplotlib Radar Chart (similar to the polar chart)
            fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
            
            # Get top 5 features for the radar chart
            top_features = feature_importance.head(5)
            categories = top_features['Feature'].tolist()
            values = top_features['Importance (%)'].tolist()
            
            # Number of variables
            N = len(categories)
            
            # Compute angle for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values
            values_plot = values.copy()
            values_plot += values[:1]  # Close the loop
            
            # Draw the plot
            ax.plot(angles, values_plot, linewidth=2, linestyle='solid')
            ax.fill(angles, values_plot, alpha=0.25)
            
            # Set category labels
            plt.xticks(angles[:-1], categories, size=10)
            
            # Set y-axis limit
            plt.ylim(0, max(values) * 1.2)
            
            # Add title
            plt.title("Feature Importance", size=12, y=1.1)
            
            # Display the radar chart
            st.pyplot(fig)
        
        # Add new row for pie chart
        st.markdown("---")
        st.subheader("Feature Importance Distribution")
        
        # Create a matplotlib pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the data for pie chart (top 6 features)
        feature_importance_top = feature_importance.head(6)
        other_importance = feature_importance.iloc[6:]['Importance (%)'].sum() if len(feature_importance) > 6 else 0
        
        # Prepare data for pie chart
        labels = feature_importance_top['Feature'].tolist()
        sizes = feature_importance_top['Importance (%)'].tolist()
        
        # Add "Other" category if needed
        if other_importance > 0:
            labels.append('Other')
            sizes.append(other_importance)
        
        # Custom colors
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
        
        # Create explode effect (slight separation for the largest slice)
        explode = [0] * len(labels)
        if sizes:  # Check if there are any sizes
            explode[sizes.index(max(sizes))] = 0.1
        
        # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode,
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90, 
            shadow=False,
            colors=colors
        )
        
        # Customize text appearance
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=10)
        
        # Equal aspect ratio ensures the pie chart is circular
        ax.axis('equal')
        
        # Add title
        plt.title('Feature Importance Distribution', fontsize=14)
        
        # Add a legend with percentages
        legend_labels = [f"{label} ({size:.1f}%)" for label, size in zip(labels, sizes)]
        plt.legend(wedges, legend_labels, title="Risk Factors", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Display the pie chart
        st.pyplot(fig)
        
        # Add interpretation text
        st.markdown("""
        ### Understanding the Results
        
        The pie chart above shows the relative importance of each factor in determining the risk level. 
        Larger slices represent factors that had a greater influence on the assessment.
        
        - **Key risk factors** are shown with their percentage contribution to the overall risk assessment
        - Addressing the top factors may help improve outcomes
        - The model considers the combined effect of all factors
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Make sure you have trained the model and generated all required model files.")