# """
# Dynamic Pricing Engine for Telematics Insurance
# Calculates premiums based on risk scores and usage patterns
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# class TelematicsPricingEngine:
#     """Comprehensive pricing engine for usage-based insurance"""
    
#     def __init__(self, base_premium=1200):
#         self.base_premium = base_premium
#         self.pricing_model = None
#         self.scaler = StandardScaler()
        
#         # Pricing rules and factors
#         self.pricing_rules = self._initialize_pricing_rules()
#         self.market_factors = self._initialize_market_factors()
        
#     def _initialize_pricing_rules(self):
#         """Initialize pricing rules and factor weights"""
#         return {
#             # Factor weights for final premium calculation
#             'risk_weight': 0.40,          # Risk score impact
#             'usage_weight': 0.25,         # Usage pattern impact
#             'behavioral_weight': 0.20,    # Behavioral factors impact
#             'demographic_weight': 0.15,   # Demographic factors impact
            
#             # Risk multipliers based on risk score percentiles
#             'risk_multipliers': {
#                 'very_low': 0.75,    # Bottom 20%
#                 'low': 0.85,         # 20-40%
#                 'medium': 1.00,      # 40-60%
#                 'high': 1.20,        # 60-80%
#                 'very_high': 1.50    # Top 20%
#             },
            
#             # Usage-based factors
#             'mileage_brackets': {
#                 5000: 0.85,      # Low mileage discount
#                 10000: 0.95,     # Moderate discount  
#                 15000: 1.00,     # Base rate
#                 20000: 1.10,     # Slight surcharge
#                 25000: 1.25,     # High mileage surcharge
#                 50000: 1.50      # Very high mileage
#             },
            
#             # Behavioral adjustments
#             'behavioral_factors': {
#                 'speeding_multiplier': 1.15,      # Per violation
#                 'harsh_braking_multiplier': 1.10, # Per event
#                 'night_driving_multiplier': 1.05,  # If > 20% night driving
#                 'weather_driving_multiplier': 0.95 # Good weather driving bonus
#             },
            
#             # Demographic factors
#             'age_factors': {
#                 '18_24': 1.40,    # Young drivers
#                 '25_34': 1.10,    # Young adults
#                 '35_49': 0.95,    # Prime age
#                 '50_64': 1.00,    # Mature drivers
#                 '65_plus': 1.15   # Senior drivers
#             },
            
#             # Vehicle factors
#             'vehicle_age_factors': {
#                 (0, 3): 0.90,     # New vehicles
#                 (3, 7): 1.00,     # Average age
#                 (7, 12): 1.10,    # Older vehicles
#                 (12, 20): 1.25    # Very old vehicles
#             },
            
#             'safety_score_bonus': 0.02,  # 2% discount per safety point above 5
            
#             # Geographic/Environmental factors
#             'environmental_factors': {
#                 'high_crime': 1.15,       # High crime index
#                 'high_accident': 1.20,    # High accident index
#                 'poor_repair': 1.10       # Poor repair cost index
#             }
#         }
    
#     def _initialize_market_factors(self):
#         """Initialize market and competitive factors"""
#         return {
#             'competitive_adjustment': 0.95,  # 5% discount vs traditional
#             'inflation_factor': 1.03,        # Annual inflation
#             'loss_ratio_target': 0.75,       # Target loss ratio
#             'profit_margin': 0.15,           # Desired profit margin
#             'regulatory_buffer': 1.05        # Regulatory compliance buffer
#         }
    
#     def calculate_usage_based_premium(self, df, risk_scores):
#         """Calculate usage-based insurance premium for each record"""
        
#         premiums = []
#         premium_breakdown = []
        
#         for idx, row in df.iterrows():
#             # Start with base premium
#             premium = self.base_premium
#             breakdown = {'base_premium': self.base_premium}
            
#             # 1. Risk Score Adjustment (40% weight)
#             risk_score = risk_scores[idx] if len(risk_scores) > idx else 50
#             risk_multiplier = self._get_risk_multiplier(risk_score)
#             risk_adjustment = premium * (risk_multiplier - 1) * self.pricing_rules['risk_weight']
#             premium += risk_adjustment
#             breakdown['risk_adjustment'] = risk_adjustment
            
#             # 2. Usage Pattern Adjustment (25% weight)
#             usage_adjustment = self._calculate_usage_adjustment(row, premium)
#             premium += usage_adjustment
#             breakdown['usage_adjustment'] = usage_adjustment
            
#             # 3. Behavioral Adjustment (20% weight)  
#             behavioral_adjustment = self._calculate_behavioral_adjustment(row, premium)
#             premium += behavioral_adjustment
#             breakdown['behavioral_adjustment'] = behavioral_adjustment
            
#             # 4. Demographic Adjustment (15% weight)
#             demographic_adjustment = self._calculate_demographic_adjustment(row, premium)
#             premium += demographic_adjustment
#             breakdown['demographic_adjustment'] = demographic_adjustment
            
#             # 5. Vehicle Adjustments
#             vehicle_adjustment = self._calculate_vehicle_adjustment(row, premium)
#             premium += vehicle_adjustment
#             breakdown['vehicle_adjustment'] = vehicle_adjustment
            
#             # 6. Environmental/Geographic Adjustments
#             environmental_adjustment = self._calculate_environmental_adjustment(row, premium)
#             premium += environmental_adjustment
#             breakdown['environmental_adjustment'] = environmental_adjustment
            
#             # 7. Apply market factors
#             for factor_name, factor_value in self.market_factors.items():
#                 if factor_name != 'loss_ratio_target' and factor_name != 'profit_margin':
#                     premium *= factor_value
            
#             # 8. Ensure reasonable bounds
#             min_premium = self.base_premium * 0.60  # Max 40% discount
#             max_premium = self.base_premium * 2.00  # Max 100% surcharge
#             premium = np.clip(premium, min_premium, max_premium)
            
#             breakdown['final_premium'] = premium
#             breakdown['total_adjustment'] = premium - self.base_premium
#             breakdown['discount_surcharge_pct'] = ((premium - self.base_premium) / self.base_premium) * 100
            
#             premiums.append(premium)
#             premium_breakdown.append(breakdown)
        
#         return np.array(premiums), premium_breakdown
    
#     def _get_risk_multiplier(self, risk_score):
#         """Get risk multiplier based on risk score percentile"""
#         if risk_score <= 20:
#             return self.pricing_rules['risk_multipliers']['very_high']
#         elif risk_score <= 40:
#             return self.pricing_rules['risk_multipliers']['high']  
#         elif risk_score <= 60:
#             return self.pricing_rules['risk_multipliers']['medium']
#         elif risk_score <= 80:
#             return self.pricing_rules['risk_multipliers']['low']
#         else:
#             return self.pricing_rules['risk_multipliers']['very_low']
    
#     def _calculate_usage_adjustment(self, row, current_premium):
#         """Calculate adjustment based on usage patterns"""
#         adjustment = 0
        
#         # Mileage adjustment
#         annual_miles = row.get('miles', 0) * 365 / 30  # Rough annualization
#         mileage_multiplier = self._get_mileage_multiplier(annual_miles)
#         mileage_adj = current_premium * (mileage_multiplier - 1) * self.pricing_rules['usage_weight']
#         adjustment += mileage_adj
        
#         # Duration-based adjustment (longer trips = more exposure)
#         avg_duration = row.get('duration_sec', 1800)  # 30 min default
#         if avg_duration > 3600:  # > 1 hour
#             adjustment += current_premium * 0.05 * self.pricing_rules['usage_weight']
#         elif avg_duration < 900:  # < 15 minutes
#             adjustment -= current_premium * 0.03 * self.pricing_rules['usage_weight']
        
#         return adjustment
    
#     def _get_mileage_multiplier(self, annual_miles):
#         """Get mileage multiplier based on annual mileage"""
#         brackets = self.pricing_rules['mileage_brackets']
#         for threshold, multiplier in sorted(brackets.items()):
#             if annual_miles <= threshold:
#                 return multiplier
#         return list(brackets.values())[-1]  # Highest bracket
    
#     def _calculate_behavioral_adjustment(self, row, current_premium):
#         """Calculate adjustment based on behavioral factors"""
#         adjustment = 0
        
#         # Speeding violations
#         speeding_events = row.get('over_limit', 0)
#         if speeding_events > 0:
#             speeding_penalty = current_premium * (self.pricing_rules['behavioral_factors']['speeding_multiplier'] - 1)
#             speeding_penalty *= min(speeding_events / 10, 1)  # Cap at 10 events
#             adjustment += speeding_penalty * self.pricing_rules['behavioral_weight']
        
#         # Harsh braking events
#         harsh_braking = row.get('harsh_brakes', 0)
#         if harsh_braking > 0:
#             braking_penalty = current_premium * (self.pricing_rules['behavioral_factors']['harsh_braking_multiplier'] - 1)
#             braking_penalty *= min(harsh_braking / 5, 1)  # Cap at 5 events
#             adjustment += braking_penalty * self.pricing_rules['behavioral_weight']
        
#         # Night driving
#         night_driving = row.get('night', 0)
#         if night_driving > 0.2:  # More than 20% night driving
#             night_penalty = current_premium * (self.pricing_rules['behavioral_factors']['night_driving_multiplier'] - 1)
#             adjustment += night_penalty * self.pricing_rules['behavioral_weight']
        
#         # Weather driving bonus
#         rain_driving = row.get('rain', 0)
#         if rain_driving < 0.1:  # Less than 10% bad weather driving
#             weather_bonus = current_premium * (1 - self.pricing_rules['behavioral_factors']['weather_driving_multiplier'])
#             adjustment -= weather_bonus * self.pricing_rules['behavioral_weight']
        
#         return adjustment
    
#     def _calculate_demographic_adjustment(self, row, current_premium):
#         """Calculate adjustment based on demographic factors"""
#         adjustment = 0
        
#         # Age-based adjustment
#         age_groups = ['age_18_24', 'age_25_34', 'age_35_49', 'age_50_64', 'age_65_plus']
#         age_factors = self.pricing_rules['age_factors']
        
#         for age_group in age_groups:
#             if row.get(age_group, 0) == 1:
#                 age_factor = age_factors.get(age_group.replace('age_', ''), 1.0)
#                 age_adj = current_premium * (age_factor - 1) * self.pricing_rules['demographic_weight']
#                 adjustment += age_adj
#                 break
        
#         # Prior claims penalty
#         prior_claims = row.get('prior_claims_count', 0)
#         if prior_claims > 0:
#             claims_penalty = current_premium * 0.15 * min(prior_claims / 3, 1)  # Cap at 3 claims
#             adjustment += claims_penalty * self.pricing_rules['demographic_weight']
        
#         # Tickets penalty
#         tickets = row.get('tickets_count', 0)
#         if tickets > 0:
#             ticket_penalty = current_premium * 0.10 * min(tickets / 2, 1)  # Cap at 2 tickets
#             adjustment += ticket_penalty * self.pricing_rules['demographic_weight']
        
#         # DUI penalty
#         if row.get('dui_flag', 0) == 1:
#             dui_penalty = current_premium * 0.50  # 50% penalty for DUI
#             adjustment += dui_penalty * self.pricing_rules['demographic_weight']
        
#         return adjustment
    
#     def _calculate_vehicle_adjustment(self, row, current_premium):
#         """Calculate adjustment based on vehicle factors"""
#         adjustment = 0
        
#         # Vehicle age adjustment
#         vehicle_age = row.get('vehicle_age', 5)
#         age_factors = self.pricing_rules['vehicle_age_factors']
        
#         for (min_age, max_age), factor in age_factors.items():
#             if min_age <= vehicle_age < max_age:
#                 age_adj = current_premium * (factor - 1) * 0.1  # 10% weight for vehicle age
#                 adjustment += age_adj
#                 break
        
#         # Safety score bonus
#         safety_score = row.get('safety_score', 5)
#         if safety_score > 5:
#             safety_bonus = current_premium * self.pricing_rules['safety_score_bonus'] * (safety_score - 5)
#             adjustment -= safety_bonus
        
#         return adjustment
    
#     def _calculate_environmental_adjustment(self, row, current_premium):
#         """Calculate adjustment based on environmental/geographic factors"""
#         adjustment = 0
        
#         # Crime index adjustment
#         crime_index = row.get('crime_index', 50)
#         if crime_index > 70:  # High crime area
#             crime_penalty = current_premium * (self.pricing_rules['environmental_factors']['high_crime'] - 1)
#             adjustment += crime_penalty * 0.05  # 5% weight
        
#         # Accident index adjustment
#         accident_index = row.get('accident_index', 100)
#         if accident_index > 130:  # High accident area
#             accident_penalty = current_premium * (self.pricing_rules['environmental_factors']['high_accident'] - 1)
#             adjustment += accident_penalty * 0.05  # 5% weight
        
#         # Repair cost adjustment
#         repair_cost_index = row.get('repair_cost_index', 1.0)
#         if repair_cost_index > 1.2:  # High repair costs
#             repair_penalty = current_premium * (self.pricing_rules['environmental_factors']['poor_repair'] - 1)
#             adjustment += repair_penalty * 0.03  # 3% weight
        
#         return adjustment
    
#     def train_ml_pricing_model(self, df, risk_scores, actual_premiums=None):
#         """Train ML model to predict optimal premiums"""
#         print("🤖 Training ML-based pricing model...")
        
#         # Prepare features for ML model
#         feature_cols = [
#             'over_limit', 'harsh_brakes', 'mean_speed', 'std_speed', 'mean_accel', 'std_accel',
#             'miles', 'duration_sec', 'night', 'rain', 'crime_index', 'accident_index',
#             'vehicle_age', 'safety_score', 'prior_claims_count', 'tickets_count', 'dui_flag',
#             'age_18_24', 'age_25_34', 'age_35_49', 'age_50_64', 'age_65_plus'
#         ]
        
#         # Add risk scores as features
#         X = df[feature_cols].copy()
#         X['risk_score'] = risk_scores
        
#         # Create engineered features
#         X['risk_exposure'] = X['miles'] * X['risk_score'] / 100
#         X['behavioral_risk'] = X['over_limit'] + X['harsh_brakes']
#         X['driver_experience'] = 1 / (1 + X['prior_claims_count'] + X['tickets_count'])
        
#         # Use rule-based premiums as targets if no actual premiums provided
#         if actual_premiums is None:
#             rule_based_premiums, _ = self.calculate_usage_based_premium(df, risk_scores)
#             y = rule_based_premiums
#         else:
#             y = actual_premiums
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Train Random Forest model
#         self.pricing_model = RandomForestRegressor(
#             n_estimators=200,
#             max_depth=12,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42
#         )
        
#         self.pricing_model.fit(X_train_scaled, y_train)
        
#         # Evaluate model
#         y_pred = self.pricing_model.predict(X_test_scaled)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)
#         mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
#         print(f"✅ ML Pricing Model Performance:")
#         print(f"   MAE: ${mae:.2f}")
#         print(f"   RMSE: ${rmse:.2f}")
#         print(f"   R²: {r2:.4f}")
#         print(f"   MAPE: {mape:.2f}%")
        
#         # Feature importance
#         feature_importance = pd.DataFrame({
#             'feature': X.columns,
#             'importance': self.pricing_model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print(f"\n🔝 Top 10 Pricing Features:")
#         print(feature_importance.head(10))
        
#         return {
#             'mae': mae,
#             'rmse': rmse,
#             'r2': r2,
#             'mape': mape,
#             'feature_importance': feature_importance
#         }
    
#     def predict_premium_ml(self, df, risk_scores):
#         """Predict premiums using trained ML model"""
#         if self.pricing_model is None:
#             raise ValueError("ML pricing model not trained yet")
        
#         # Prepare features (same as training)
#         feature_cols = [
#             'over_limit', 'harsh_brakes', 'mean_speed', 'std_speed', 'mean_accel', 'std_accel',
#             'miles', 'duration_sec', 'night', 'rain', 'crime_index', 'accident_index',
#             'vehicle_age', 'safety_score', 'prior_claims_count', 'tickets_count', 'dui_flag',
#             'age_18_24', 'age_25_34', 'age_35_49', 'age_50_64', 'age_65_plus'
#         ]
        
#         X = df[feature_cols].copy()
#         X['risk_score'] = risk_scores
        
#         # Add engineered features
#         X['risk_exposure'] = X['miles'] * X['risk_score'] / 100
#         X['behavioral_risk'] = X['over_limit'] + X['harsh_brakes']
#         X['driver_experience'] = 1 / (1 + X['prior_claims_count'] + X['tickets_count'])
        
#         # Scale and predict
#         X_scaled = self.scaler.transform(X)
#         ml_premiums = self.pricing_model.predict(X_scaled)
        
#         return ml_premiums
    
#     def analyze_pricing_performance(self, df, risk_scores, actual_claims=None, actual_severity=None):
#         """Analyze pricing model performance and effectiveness"""
        
#         # Calculate rule-based and ML premiums
#         rule_premiums, premium_breakdown = self.calculate_usage_based_premium(df, risk_scores)
        
#         if self.pricing_model is not None:
#             ml_premiums = self.predict_premium_ml(df, risk_scores)
#         else:
#             ml_premiums = rule_premiums
        
#         # Create analysis dataframe
#         analysis_df = pd.DataFrame({
#             'risk_score': risk_scores,
#             'rule_premium': rule_premiums,
#             'ml_premium': ml_premiums,
#             'base_premium': self.base_premium,
#             'actual_claim': actual_claims if actual_claims is not None else df.get('claim', 0),
#             'actual_severity': actual_severity if actual_severity is not None else df.get('severity', 0)
#         })
        
#         # Risk-premium alignment analysis
#         correlation_rule = np.corrcoef(analysis_df['risk_score'], analysis_df['rule_premium'])[0, 1]
#         correlation_ml = np.corrcoef(analysis_df['risk_score'], analysis_df['ml_premium'])[0, 1]
        
#         # Premium distribution by risk segments
#         analysis_df['risk_segment'] = pd.cut(
#             analysis_df['risk_score'], 
#             bins=[0, 20, 40, 60, 80, 100],
#             labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
#         )
        
#         segment_analysis = analysis_df.groupby('risk_segment').agg({
#             'risk_score': 'mean',
#             'rule_premium': 'mean',
#             'ml_premium': 'mean',
#             'base_premium': 'mean',
#             'actual_claim': 'mean'
#         }).round(2)
        
#         # Calculate loss ratios by segment (if claims data available)
#         if actual_claims is not None and actual_severity is not None:
#             analysis_df['expected_loss'] = analysis_df['actual_claim'] * analysis_df['actual_severity']
#             segment_analysis['expected_loss'] = analysis_df.groupby('risk_segment')['expected_loss'].mean()
#             segment_analysis['rule_loss_ratio'] = segment_analysis['expected_loss'] / segment_analysis['rule_premium']
#             segment_analysis['ml_loss_ratio'] = segment_analysis['expected_loss'] / segment_analysis['ml_premium']
        
#         results = {
#             'risk_premium_correlation': {
#                 'rule_based': correlation_rule,
#                 'ml_based': correlation_ml
#             },
#             'segment_analysis': segment_analysis,
#             'premium_statistics': {
#                 'rule_based': {
#                     'mean': analysis_df['rule_premium'].mean(),
#                     'std': analysis_df['rule_premium'].std(),
#                     'min': analysis_df['rule_premium'].min(),
#                     'max': analysis_df['rule_premium'].max()
#                 },
#                 'ml_based': {
#                     'mean': analysis_df['ml_premium'].mean(),
#                     'std': analysis_df['ml_premium'].std(),
#                     'min': analysis_df['ml_premium'].min(),
#                     'max': analysis_df['ml_premium'].max()
#                 }
#             }
#         }
        
#         return results, analysis_df
    
#     def plot_pricing_analysis(self, analysis_df):
#         """Plot pricing analysis results"""
#         fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
#         # Plot 1: Risk Score vs Premium (Rule-based)
#         axes[0, 0].scatter(analysis_df['risk_score'], analysis_df['rule_premium'], alpha=0.6)
#         axes[0, 0].set_xlabel('Risk Score')
#         axes[0, 0].set_ylabel('Rule-Based Premium ($)')
#         axes[0, 0].set_title('Risk Score vs Rule-Based Premium')
        
#         # Plot 2: Risk Score vs Premium (ML-based)
#         axes[0, 1].scatter(analysis_df['risk_score'], analysis_df['ml_premium'], alpha=0.6, color='orange')
#         axes[0, 1].set_xlabel('Risk Score')
#         axes[0, 1].set_ylabel('ML-Based Premium ($)')
#         axes[0, 1].set_title('Risk Score vs ML-Based Premium')
        
#         # Plot 3: Premium Distribution by Risk Segment
#         segment_premiums = analysis_df.groupby('risk_segment')[['rule_premium', 'ml_premium']].mean()
#         segment_premiums.plot(kind='bar', ax=axes[0, 2])
#         axes[0, 2].set_title('Average Premium by Risk Segment')
#         axes[0, 2].set_ylabel('Premium ($)')
#         axes[0, 2].tick_params(axis='x', rotation=45)
        
#         # Plot 4: Premium Distribution Histogram
#         axes[1, 0].hist(analysis_df['rule_premium'], bins=50, alpha=0.7, label='Rule-based', color='blue')
#         axes[1, 0].hist(analysis_df['ml_premium'], bins=50, alpha=0.7, label='ML-based', color='orange')
#         axes[1, 0].set_xlabel('Premium ($)')
#         axes[1, 0].set_ylabel('Frequency')
#         axes[1, 0].set_title('Premium Distribution Comparison')
#         axes[1, 0].legend()
        
#         # Plot 5: Rule vs ML Premium Comparison
#         axes[1, 1].scatter(analysis_df['rule_premium'], analysis_df['ml_premium'], alpha=0.6)
#         axes[1, 1].plot([analysis_df['rule_premium'].min(), analysis_df['rule_premium'].max()], 
#                        [analysis_df['rule_premium'].min(), analysis_df['rule_premium'].max()], 
#                        'r--', label='Perfect Agreement')
#         axes[1, 1].set_xlabel('Rule-Based Premium ($)')
#         axes[1, 1].set_ylabel('ML-Based Premium ($)')
#         axes[1, 1].set_title('Rule-Based vs ML-Based Premium')
#         axes[1, 1].legend()
        
#         # Plot 6: Premium Savings Distribution
#         analysis_df['savings'] = self.base_premium - analysis_df['rule_premium']
#         axes[1, 2].hist(analysis_df['savings'], bins=50, alpha=0.7, color='green')
#         axes[1, 2].axvline(analysis_df['savings'].mean(), color='red', linestyle='--', 
#                           label=f'Mean Savings: ${analysis_df["savings"].mean():.2f}')
#         axes[1, 2].set_xlabel('Premium Savings ($)')
#         axes[1, 2].set_ylabel('Frequency')
#         axes[1, 2].set_title('Premium Savings Distribution')
#         axes[1, 2].legend()
        
#         plt.tight_layout()
#         plt.show()
    
#     def save_pricing_models(self, path='models/'):
#         """Save pricing models and components"""
#         import os
#         os.makedirs(path, exist_ok=True)
        
#         # Save ML model if trained
#         if self.pricing_model is not None:
#             joblib.dump(self.pricing_model, f'{path}/pricing_model.pkl')
#             joblib.dump(self.scaler, f'{path}/pricing_scaler.pkl')
#             print(f"✅ ML pricing model saved to {path}")
        
#         # Save pricing rules and configuration
#         joblib.dump(self.pricing_rules, f'{path}/pricing_rules.pkl')
#         joblib.dump(self.market_factors, f'{path}/market_factors.pkl')
        
#         # Save base premium
#         config = {
#             'base_premium': self.base_premium,
#             'model_version': '1.0.0',
#             'last_updated': datetime.now().isoformat()
#         }
#         joblib.dump(config, f'{path}/pricing_config.pkl')
        
#         print(f"✅ Pricing configuration saved to {path}")
    
#     def load_pricing_models(self, path='models/'):
#         """Load pricing models and components"""
#         try:
#             # Load ML model if exists
#             try:
#                 self.pricing_model = joblib.load(f'{path}/pricing_model.pkl')
#                 self.scaler = joblib.load(f'{path}/pricing_scaler.pkl')
#                 print(f"✅ ML pricing model loaded from {path}")
#             except FileNotFoundError:
#                 print("ℹ️  No ML pricing model found, using rule-based only")
            
#             # Load pricing configuration
#             self.pricing_rules = joblib.load(f'{path}/pricing_rules.pkl')
#             self.market_factors = joblib.load(f'{path}/market_factors.pkl')
#             config = joblib.load(f'{path}/pricing_config.pkl')
#             self.base_premium = config['base_premium']
            
#             print(f"✅ Pricing configuration loaded from {path}")
#         except FileNotFoundError as e:
#             print(f"❌ Error loading pricing models: {e}")

# def main():
#     """Main pricing engine demonstration"""
    
#     # Load data (using same sample data as risk model)
#     print("📊 Generating sample data for pricing analysis...")
    
#     # Use the same data generation as in risk model
#     np.random.seed(42)
#     n_samples = 10000
    
#     sample_data = {
#         'over_limit': np.random.poisson(2, n_samples),
#         'harsh_brakes': np.random.poisson(1, n_samples),
#         'mean_speed': np.random.normal(45, 10, n_samples),
#         'std_speed': np.random.gamma(2, 5, n_samples),
#         'mean_accel': np.random.normal(0, 0.5, n_samples),
#         'std_accel': np.random.gamma(1, 0.3, n_samples),
#         'miles': np.random.gamma(2, 20, n_samples),
#         'duration_sec': np.random.gamma(3, 600, n_samples),
#         'night': np.random.binomial(1, 0.2, n_samples),
#         'rain': np.random.binomial(1, 0.15, n_samples),
#         'crime_index': np.random.normal(50, 20, n_samples),
#         'accident_index': np.random.normal(100, 30, n_samples),
#         'repair_cost_index': np.random.normal(1, 0.3, n_samples),
#         'parts_availability_index': np.random.normal(1, 0.2, n_samples),
#         'age': np.random.normal(40, 15, n_samples),
#         'age_18_24': np.random.binomial(1, 0.1, n_samples),
#         'age_25_34': np.random.binomial(1, 0.25, n_samples),
#         'age_35_49': np.random.binomial(1, 0.35, n_samples),
#         'age_50_64': np.random.binomial(1, 0.25, n_samples),
#         'age_65_plus': np.random.binomial(1, 0.05, n_samples),
#         'gender_male': np.random.binomial(1, 0.5, n_samples),
#         'gender_female': np.random.binomial(1, 0.48, n_samples),
#         'gender_nonbinary': np.random.binomial(1, 0.01, n_samples),
#         'gender_unknown': np.random.binomial(1, 0.01, n_samples),
#         'height_m': np.random.normal(1.7, 0.1, n_samples),
#         'num_drivers': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
#         'prior_claims_count': np.random.poisson(0.5, n_samples),
#         'tickets_count': np.random.poisson(0.3, n_samples),
#         'dui_flag': np.random.binomial(1, 0.03, n_samples),
#         'vehicle_age': np.random.gamma(2, 3, n_samples),
#         'safety_score': np.random.normal(7, 2, n_samples),
#         'annual_miles_gap': np.random.normal(0, 5000, n_samples),
#         'primary_use_commute': np.random.binomial(1, 0.6, n_samples),
#         'primary_use_business': np.random.binomial(1, 0.1, n_samples),
#     }
    
#     df = pd.DataFrame(sample_data)
    
#     # Generate realistic risk scores (simulated)
#     risk_scores = (
#         50 + 
#         -10 * df['over_limit'] / df['over_limit'].max() +
#         -15 * df['harsh_brakes'] / df['harsh_brakes'].max() +
#         -5 * df['prior_claims_count'] +
#         -20 * df['dui_flag'] +
#         10 * df['safety_score'] / 10 +
#         np.random.normal(0, 10, n_samples)
#     )
#     risk_scores = np.clip(risk_scores, 0, 100)
    
#     # Generate claims data
#     claim_logits = (
#         0.1 * df['over_limit'] +
#         0.15 * df['harsh_brakes'] +
#         0.2 * df['prior_claims_count'] +
#         0.3 * df['dui_flag'] +
#         -0.05 * df['safety_score'] - 2
#     )
#     df['claim'] = np.random.binomial(1, 1 / (1 + np.exp(-claim_logits)))
    
#     severity_base = 3000 + 1000 * df['over_limit'] + 500 * df['harsh_brakes']
#     df['severity'] = np.where(df['claim'] == 1, np.random.gamma(2, severity_base / 2), 0)
    
#     print(f"✅ Dataset created: {df.shape}")
#     print(f"📊 Claim rate: {df['claim'].mean():.2%}")
#     print(f"💰 Average severity (claims only): ${df[df['claim']==1]['severity'].mean():.2f}")
    
#     # Initialize pricing engine
#     pricing_engine = TelematicsPricingEngine(base_premium=1200)
    
#     # Calculate rule-based premiums
#     print("\n💰 Calculating rule-based premiums...")
#     rule_premiums, premium_breakdown = pricing_engine.calculate_usage_based_premium(df, risk_scores)
    
#     print(f"✅ Rule-based premiums calculated")
#     print(f"   Average premium: ${rule_premiums.mean():.2f}")
#     print(f"   Premium range: ${rule_premiums.min():.2f} - ${rule_premiums.max():.2f}")
#     print(f"   Average savings vs base: ${pricing_engine.base_premium - rule_premiums.mean():.2f}")
    
#     # Train ML pricing model
#     ml_metrics = pricing_engine.train_ml_pricing_model(df, risk_scores)
    
#     # Analyze pricing performance
#     print("\n📊 Analyzing pricing performance...")
#     pricing_results, analysis_df = pricing_engine.analyze_pricing_performance(
#         df, risk_scores, df['claim'], df['severity']
#     )
    
#     print("\n🎯 Pricing Analysis Results:")
#     print("Risk-Premium Correlations:")
#     for method, corr in pricing_results['risk_premium_correlation'].items():
#         print(f"  {method}: {corr:.4f}")
    
#     print("\n📊 Premium Statistics by Risk Segment:")
#     print(pricing_results['segment_analysis'])
    
#     # Plot results
#     pricing_engine.plot_pricing_analysis(analysis_df)
    
#     # Save models
#     pricing_engine.save_pricing_models()
    
#     print("\n✅ Pricing engine training complete!")
    
#     return pricing_engine, analysis_df

# if __name__ == "__main__":
#     pricing_engine, analysis_df = main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Pricing Engine for Telematics Insurance
- Rule-based premiums with market & risk factors
- Optional calibration to target loss ratio (if expected_loss available)
- ML pricing model with GridSearchCV (XGB / RF / GBRT)
- Visuals: alignment, segments, distributions, lift & cumulative curves

Usage:
  python src/pricing_engine.py \
    --features data/features.csv \
    --trip_scores data/trip_scores.csv \
    --out_dir models_pricing \
    --base_premium 1200
"""

# from __future__ import annotations
# import argparse, json, warnings
# from pathlib import Path
# from typing import Tuple, Dict, List, Optional

# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# import xgboost as xgb
# import joblib

# warnings.filterwarnings("ignore")


# # -----------------------------
# # Helpers: IO & dataset creation
# # -----------------------------
# def load_inputs(features_path: str, scores_path: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
#     if not Path(features_path).exists():
#         raise SystemExit(f"[FATAL] features file not found: {features_path}")
#     feats = pd.read_csv(features_path)
#     scores = None
#     if scores_path and Path(scores_path).exists():
#         scores = pd.read_csv(scores_path)
#     return feats, scores


# def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         for m in missing:
#             df[m] = 0.0
#     return df


# # -----------------------------
# # Pricing Engine
# # -----------------------------
# class TelematicsPricingEngine:
#     """
#     Comprehensive pricing engine for usage-based insurance.
#     NOTE: Uses your pipeline semantics:
#       - over_limit, harsh_brakes, night, rain are FRACTIONS (0..1)
#       - miles ~ per-trip miles, duration_sec ~ per-trip duration
#     """

#     def __init__(self, base_premium: float = 1200.0):
#         self.base_premium = float(base_premium)
#         self.scaler = StandardScaler()
#         self.pricing_model = None
#         self.pricing_rules = self._init_pricing_rules()
#         self.market_factors = self._init_market_factors()
#         self.quantiles_: Dict[str, float] = {}

#     def _init_pricing_rules(self) -> dict:
#         return {
#             # high-level weights into final premium delta
#             "risk_weight": 0.40,
#             "usage_weight": 0.25,
#             "behavior_weight": 0.20,
#             "demo_weight": 0.15,

#             # nominal risk multipliers; mapping later uses dataset deciles unless fixed bands requested
#             "risk_multipliers": {
#                 "very_low": 0.75,   # top deciles (low risk)
#                 "low": 0.85,
#                 "medium": 1.00,
#                 "high": 1.20,
#                 "very_high": 1.50,  # bottom deciles (high risk)
#             },

#             # annual mileage brackets (rough annualization from recent miles)
#             "mileage_brackets": {
#                 5000: 0.88,
#                 10000: 0.96,
#                 15000: 1.00,
#                 20000: 1.08,
#                 30000: 1.18,
#                 50000: 1.35
#             },

#             # behavior multipliers for FRACTIONS (0..1), scaled internally
#             "behavior": {
#                 "speeding_max_mult": 1.18,      # applied when over_limit≈1
#                 "braking_max_mult": 1.12,       # applied when harsh_brakes≈1
#                 "night_thresh": 0.20,
#                 "night_mult": 1.05,             # if night share>thresh
#                 "good_weather_bonus": 0.97      # if rain share < 0.10
#             },

#             # demographics (one-hot groups)
#             "age_factors": {  # multiplicative
#                 "age_18_24": 1.40,
#                 "age_25_34": 1.10,
#                 "age_35_49": 0.95,
#                 "age_50_64": 1.00,
#                 "age_65_plus": 1.15,
#             },

#             # vehicle
#             "vehicle_age_factors": [
#                 ((0, 3), 0.92),
#                 ((3, 7), 1.00),
#                 ((7, 12), 1.08),
#                 ((12, 20), 1.18),
#                 ((20, 100), 1.28),
#             ],
#             "safety_score_bonus_pct_per_point_above5": 0.02,  # 2% discount per safety point >5

#             # environment
#             "env": {
#                 "crime_hi_thresh": 70,
#                 "crime_mult": 1.12,
#                 "accident_hi_thresh": 130,
#                 "accident_mult": 1.15,
#                 "repair_hi_thresh": 1.20,
#                 "repair_mult": 1.08,
#             },

#             # guardrails
#             "max_discount": 0.45,  # 45% off base
#             "max_surcharge": 1.10, # 110% over base
#         }

#     def _init_market_factors(self) -> dict:
#         return {
#             "competitive_adjustment": 0.97,
#             "inflation_factor": 1.03,
#             "regulatory_buffer": 1.02,
#             "loss_ratio_target": 0.75,
#             "profit_margin": 0.15,
#         }

#     # ---------- Quantiles & Risk bucket ----------
#     def fit_quantiles(self, risk_scores: np.ndarray):
#         # compute deciles to map 0..100 risks into buckets (high risk = low score)
#         self.quantiles_ = {
#             "p20": float(np.nanpercentile(risk_scores, 20)),
#             "p40": float(np.nanpercentile(risk_scores, 40)),
#             "p60": float(np.nanpercentile(risk_scores, 60)),
#             "p80": float(np.nanpercentile(risk_scores, 80)),
#         }

#     def _risk_multiplier_from_score(self, s: float) -> float:
#         # lower score => riskier
#         r = self.pricing_rules["risk_multipliers"]
#         q = self.quantiles_
#         if not q:  # fall back to static bands if not fit
#             if s <= 20:   return r["very_high"]
#             if s <= 40:   return r["high"]
#             if s <= 60:   return r["medium"]
#             if s <= 80:   return r["low"]
#             return r["very_low"]

#         if s <= q["p20"]:  return r["very_high"]
#         if s <= q["p40"]:  return r["high"]
#         if s <= q["p60"]:  return r["medium"]
#         if s <= q["p80"]:  return r["low"]
#         return r["very_low"]

#     # ---------- Rule-based pricing ----------
#     def _usage_adjust(self, row: pd.Series, current: float) -> float:
#         # annualize recent miles (simple: last period * 365/30)
#         miles = float(row.get("miles", 0.0))
#         annual_miles = miles * 365.0 / 30.0
#         mb = self.pricing_rules["mileage_brackets"]
#         mm = list(mb.items())
#         mm.sort(key=lambda x: x[0])
#         mult = mm[-1][1]
#         for th, m in mm:
#             if annual_miles <= th:
#                 mult = m; break
#         # duration: longer avg duration => exposure
#         dur = float(row.get("duration_sec", 1800.0))
#         dur_adj = 0.0
#         if dur > 3600:      dur_adj = 0.05
#         elif dur < 900:     dur_adj = -0.03
#         return current * ((mult - 1.0) + dur_adj)

#     def _behavior_adjust(self, row: pd.Series, current: float) -> float:
#         b = self.pricing_rules["behavior"]
#         adj = 0.0
#         # over_limit & harsh_brakes are FRACTIONS
#         over = float(row.get("over_limit", 0.0))
#         hb = float(row.get("harsh_brakes", 0.0))
#         # scale each linearly 0..1 into 1..max_mult, then apply weight downstream
#         if over > 0:
#             adj += current * ( (1 + (b["speeding_max_mult"]-1)*over) - 1 )
#         if hb > 0:
#             adj += current * ( (1 + (b["braking_max_mult"]-1)*hb) - 1 )
#         # night share
#         if float(row.get("night", 0.0)) > b["night_thresh"]:
#             adj += current * (b["night_mult"] - 1)
#         # good-weather bonus: if rain fraction small
#         if float(row.get("rain", 0.0)) < 0.10:
#             adj += current * (b["good_weather_bonus"] - 1)
#         return adj

#     def _demo_adjust(self, row: pd.Series, current: float) -> float:
#         agef = self.pricing_rules["age_factors"]
#         demo_adj = 0.0
#         for k, mult in agef.items():
#             if int(row.get(k, 0)) == 1:
#                 demo_adj += current * (mult - 1.0)
#                 break  # one bucket
#         # Prior history
#         prior_claims = float(row.get("prior_claims_count", 0.0))
#         tickets = float(row.get("tickets_count", 0.0))
#         dui = int(row.get("dui_flag", 0))
#         if prior_claims > 0:  demo_adj += current * 0.15 * min(prior_claims/3.0, 1.0)
#         if tickets > 0:       demo_adj += current * 0.10 * min(tickets/2.0, 1.0)
#         if dui == 1:          demo_adj += current * 0.50
#         return demo_adj

#     def _vehicle_adjust(self, row: pd.Series, current: float) -> float:
#         vage = float(row.get("vehicle_age", 5.0))
#         fac = 0.0
#         for (lo, hi), mult in self.pricing_rules["vehicle_age_factors"]:
#             if lo <= vage < hi:
#                 fac += current * (mult - 1.0) * 0.10  # give small weight (vehicle dimension)
#                 break
#         safety = float(row.get("safety_score", 5.0))
#         if safety > 5:
#             bonus_pct = self.pricing_rules["safety_score_bonus_pct_per_point_above5"] * (safety - 5.0)
#             fac -= current * bonus_pct
#         return fac

#     def _env_adjust(self, row: pd.Series, current: float) -> float:
#         e = self.pricing_rules["env"]
#         adj = 0.0
#         if float(row.get("crime_index", 50)) > e["crime_hi_thresh"]:
#             adj += current * (e["crime_mult"] - 1) * 0.05
#         if float(row.get("accident_index", 100)) > e["accident_hi_thresh"]:
#             adj += current * (e["accident_mult"] - 1) * 0.05
#         if float(row.get("repair_cost_index", 1.0)) > e["repair_hi_thresh"]:
#             adj += current * (e["repair_mult"] - 1) * 0.03
#         return adj

#     def price_rule_based(self, df: pd.DataFrame, risk_scores: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
#         """
#         Rule-based pricing with market factors & guardrails.
#         risk_scores: 0..100 (higher = safer).
#         """
#         if self.quantiles_ == {}:
#             self.fit_quantiles(risk_scores)

#         premiums, details = [], []
#         for i, row in df.iterrows():
#             p = self.base_premium
#             breakdown = {"base": self.base_premium}

#             # 1) risk bucket
#             rmult = self._risk_multiplier_from_score(risk_scores[i])
#             adj = p * (rmult - 1) * self.pricing_rules["risk_weight"]
#             p += adj; breakdown["risk_adj"] = adj

#             # 2) usage
#             adj = self._usage_adjust(row, p) * self.pricing_rules["usage_weight"]
#             p += adj; breakdown["usage_adj"] = adj

#             # 3) behavior
#             adj = self._behavior_adjust(row, p) * self.pricing_rules["behavior_weight"]
#             p += adj; breakdown["behavior_adj"] = adj

#             # 4) demographics
#             adj = self._demo_adjust(row, p) * self.pricing_rules["demo_weight"]
#             p += adj; breakdown["demo_adj"] = adj

#             # 5) vehicle
#             adj = self._vehicle_adjust(row, p)
#             p += adj; breakdown["vehicle_adj"] = adj

#             # 6) environment
#             adj = self._env_adjust(row, p)
#             p += adj; breakdown["env_adj"] = adj

#             # 7) market factors (exclude targets)
#             p *= self.market_factors["competitive_adjustment"]
#             p *= self.market_factors["inflation_factor"]
#             p *= self.market_factors["regulatory_buffer"]

#             # 8) guardrails
#             lo = self.base_premium * (1 - self.pricing_rules["max_discount"])
#             hi = self.base_premium * (1 + self.pricing_rules["max_surcharge"])
#             p = float(np.clip(p, lo, hi))
#             breakdown["final"] = p
#             premiums.append(p); details.append(breakdown)

#         return np.array(premiums), details

#     # ---------- Loss-ratio calibration (optional) ----------
#     def calibrate_to_loss_ratio(self, premiums: np.ndarray, expected_loss: np.ndarray) -> np.ndarray:
#         """
#         Scale premiums to hit target portfolio loss ratio.
#         """
#         if expected_loss is None or np.all(~np.isfinite(expected_loss)):
#             return premiums
#         lr = float(np.sum(expected_loss) / np.sum(premiums))
#         target = self.market_factors["loss_ratio_target"]
#         if lr <= 0 or not np.isfinite(lr):
#             return premiums
#         scale = lr / target  # if lr>target => scale>1 => increase premium
#         return premiums * scale

#     # ---------- ML pricing model with tuning ----------
#     def train_ml_pricer(self, df: pd.DataFrame, risk_scores: np.ndarray,
#                         target_premiums: Optional[np.ndarray] = None) -> Dict[str, float]:
#         """
#         If target premiums not provided, uses rule-based output as pseudo-labels.
#         """
#         feat_cols = [
#             'over_limit','harsh_brakes','mean_speed','std_speed','mean_accel','std_accel',
#             'miles','duration_sec','night','rain','crime_index','accident_index',
#             'vehicle_age','safety_score','prior_claims_count','tickets_count','dui_flag',
#             'age_18_24','age_25_34','age_35_49','age_50_64','age_65_plus'
#         ]
#         df = ensure_columns(df.copy(), feat_cols)
#         X = df[feat_cols].astype(float).copy()
#         X["risk_score"] = np.asarray(risk_scores, dtype=float)
#         # engineered
#         X["risk_exposure"] = X["miles"] * (X["risk_score"] / 100.0)
#         X["behavioral_risk"] = X["over_limit"] + X["harsh_brakes"]
#         X["driver_experience"] = 1.0 / (1.0 + X["prior_claims_count"] + X["tickets_count"])

#         if target_premiums is None:
#             rule_prem, _ = self.price_rule_based(df, risk_scores)
#             y = rule_prem
#         else:
#             y = np.asarray(target_premiums, dtype=float)

#         Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#         Xtr_s = self.scaler.fit_transform(Xtr)
#         Xte_s = self.scaler.transform(Xte)

#         candidates = {
#             "XGBRegressor": (
#                 xgb.XGBRegressor(tree_method="hist", n_jobs=0, random_state=42),
#                 {
#                     "n_estimators": [600, 900],
#                     "max_depth": [4, 6],
#                     "learning_rate": [0.05, 0.1],
#                     "subsample": [0.8, 1.0],
#                     "colsample_bytree": [0.8, 1.0],
#                 },
#             ),
#             "RandomForestRegressor": (
#                 RandomForestRegressor(random_state=42, n_jobs=-1),
#                 {
#                     "n_estimators": [600, 900],
#                     "max_depth": [None, 20],
#                     "min_samples_leaf": [1, 4],
#                 },
#             ),
#             "GradientBoostingRegressor": (
#                 GradientBoostingRegressor(random_state=42),
#                 {
#                     "n_estimators": [600, 900],
#                     "learning_rate": [0.05, 0.1],
#                     "max_depth": [2, 3],
#                 },
#             ),
#         }

#         best_mae = np.inf; best_est=None; best_name=""
#         cv = KFold(n_splits=5, shuffle=True, random_state=42)
#         for name, (est, grid) in candidates.items():
#             gs = GridSearchCV(est, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True, verbose=0)
#             gs.fit(Xtr_s, ytr)
#             mae = -gs.best_score_
#             print(f"[ML][{name}] CV best MAE={mae:.2f} params={gs.best_params_}")
#             if mae < best_mae:
#                 best_mae = mae; best_est = gs.best_estimator_; best_name = name

#         self.pricing_model = best_est
#         ypred = self.pricing_model.predict(Xte_s)
#         mae = mean_absolute_error(yte, ypred)
#         rmse = mean_squared_error(yte, ypred, squared=False)
#         r2 = r2_score(yte, ypred)
#         print(f"[ML][{best_name}] Holdout MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")

#         fi = None
#         if hasattr(self.pricing_model, "feature_importances_"):
#             fi = pd.DataFrame({"feature": X.columns, "importance": self.pricing_model.feature_importances_}) \
#                 .sort_values("importance", ascending=False)
#             print("\nTop pricing features:\n", fi.head(12).to_string(index=False))

#         self._ml_train_cols = list(X.columns)  # remember exact train columns
#         return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

#     def predict_ml(self, df: pd.DataFrame, risk_scores: np.ndarray) -> np.ndarray:
#         if self.pricing_model is None:
#             raise ValueError("ML pricing model not trained.")
#         X = df.copy()
#         cols = [
#             'over_limit','harsh_brakes','mean_speed','std_speed','mean_accel','std_accel',
#             'miles','duration_sec','night','rain','crime_index','accident_index',
#             'vehicle_age','safety_score','prior_claims_count','tickets_count','dui_flag',
#             'age_18_24','age_25_34','age_35_49','age_50_64','age_65_plus'
#         ]
#         X = ensure_columns(X, cols)
#         X = X[cols].astype(float)
#         X["risk_score"] = np.asarray(risk_scores, dtype=float)
#         X["risk_exposure"] = X["miles"] * (X["risk_score"] / 100.0)
#         X["behavioral_risk"] = X["over_limit"] + X["harsh_brakes"]
#         X["driver_experience"] = 1.0 / (1.0 + X["prior_claims_count"] + X["tickets_count"])
#         # align to training columns
#         X = X.reindex(columns=self._ml_train_cols, fill_value=0.0)
#         Xs = self.scaler.transform(X)
#         return self.pricing_model.predict(Xs)

#     # ---------- Analysis & Visuals ----------
#     @staticmethod
#     def analyze(df: pd.DataFrame, risk: np.ndarray, prem_rule: np.ndarray, prem_ml: np.ndarray,
#                 actual_claim: Optional[np.ndarray] = None, severity: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Dict]:
#         out = pd.DataFrame({
#             "risk_score": risk,
#             "premium_rule": prem_rule,
#             "premium_ml": prem_ml,
#         })
#         if actual_claim is not None:
#             out["claim"] = np.asarray(actual_claim, dtype=float)
#         if severity is not None:
#             out["severity"] = np.asarray(severity, dtype=float)
#             out["loss"] = out.get("claim", 0.0) * out["severity"]
#         out["segment"] = pd.qcut(out["risk_score"], q=5, labels=["Very High","High","Medium","Low","Very Low"])

#         seg = out.groupby("segment", observed=True).agg(
#             risk=("risk_score","mean"),
#             prem_rule=("premium_rule","mean"),
#             prem_ml=("premium_ml","mean"),
#             cnt=("risk_score","size"),
#             loss=("loss","mean") if "loss" in out else ("risk_score","mean"),
#             claim_rate=("claim","mean") if "claim" in out else ("risk_score","mean")
#         ).reset_index()

#         metrics = {
#             "corr_rule": float(np.corrcoef(out["risk_score"], out["premium_rule"])[0,1]),
#             "corr_ml": float(np.corrcoef(out["risk_score"], out["premium_ml"])[0,1]),
#         }
#         return out, {"segments": seg, "metrics": metrics}

#     @staticmethod
#     def plot_all(analysis_df: pd.DataFrame, base_premium: float):
#         sns.set_style("whitegrid")
#         fig, axes = plt.subplots(2, 3, figsize=(18, 11))

#         # 1) Risk vs rule premium
#         axes[0,0].scatter(analysis_df["risk_score"], analysis_df["premium_rule"], s=8, alpha=0.4)
#         axes[0,0].axhline(base_premium, color='k', ls='--', lw=1)
#         axes[0,0].set(title="Risk vs Rule Premium", xlabel="Risk score (0-100, higher safer)", ylabel="Rule premium")

#         # 2) Risk vs ML premium
#         axes[0,1].scatter(analysis_df["risk_score"], analysis_df["premium_ml"], s=8, alpha=0.4, color="orange")
#         axes[0,1].axhline(base_premium, color='k', ls='--', lw=1)
#         axes[0,1].set(title="Risk vs ML Premium", xlabel="Risk score", ylabel="ML premium")

#         # 3) Segment averages
#         seg = analysis_df.groupby("segment")[["premium_rule","premium_ml"]].mean()
#         seg.plot(kind="bar", ax=axes[0,2])
#         axes[0,2].set(title="Avg premium by risk segment", ylabel="Premium"); axes[0,2].tick_params(axis='x', rotation=45)

#         # 4) Distributions
#         axes[1,0].hist(analysis_df["premium_rule"], bins=40, alpha=0.7, label="Rule")
#         axes[1,0].hist(analysis_df["premium_ml"], bins=40, alpha=0.7, label="ML")
#         axes[1,0].legend(); axes[1,0].set(title="Premium distribution", xlabel="Premium", ylabel="Count")

#         # 5) Rule vs ML scatter
#         mn = min(analysis_df["premium_rule"].min(), analysis_df["premium_ml"].min())
#         mx = max(analysis_df["premium_rule"].max(), analysis_df["premium_ml"].max())
#         axes[1,1].scatter(analysis_df["premium_rule"], analysis_df["premium_ml"], s=8, alpha=0.4)
#         axes[1,1].plot([mn, mx], [mn, mx], 'r--', lw=1)
#         axes[1,1].set(title="Rule vs ML premium", xlabel="Rule", ylabel="ML")

#         # 6) Cumulative premium & loss (if available)
#         try:
#             dfc = analysis_df.copy().sort_values("risk_score")  # riskier first
#             dfc["cum_prem_rule"] = dfc["premium_rule"].cumsum() / dfc["premium_rule"].sum()
#             dfc["cum_prem_ml"] = dfc["premium_ml"].cumsum() / dfc["premium_ml"].sum()
#             if "loss" in dfc:
#                 dfc["cum_loss"] = dfc["loss"].cumsum() / max(dfc["loss"].sum(), 1e-9)
#                 axes[1,2].plot(dfc["cum_loss"].values, label="Cum loss")
#             axes[1,2].plot(dfc["cum_prem_rule"].values, label="Cum prem (rule)")
#             axes[1,2].plot(dfc["cum_prem_ml"].values, label="Cum prem (ML)")
#             axes[1,2].set(title="Cumulative curves (sorted by riskier→safer)",
#                           xlabel="Policies (cumulative share)", ylabel="Cumulative share")
#             axes[1,2].legend()
#         except Exception:
#             axes[1,2].text(0.5,0.5,"Not enough data for cumulative plot", ha="center")

#         plt.tight_layout(); plt.show()

#     # ---------- Persistence ----------
#     def save(self, out_dir: str):
#         p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
#         joblib.dump(self.pricing_model, p/"pricing_model.pkl")
#         joblib.dump(self.scaler, p/"pricing_scaler.pkl")
#         with open(p/"pricing_rules.json","w") as f: json.dump(self.pricing_rules, f, indent=2)
#         with open(p/"market_factors.json","w") as f: json.dump(self.market_factors, f, indent=2)
#         meta = {"base_premium": self.base_premium, "train_cols": getattr(self, "_ml_train_cols", [])}
#         with open(p/"meta.json","w") as f: json.dump(meta, f, indent=2)
#         print(f"[SAVED] Pricing artifacts -> {p.resolve()}")

#     def load(self, out_dir: str):
#         p = Path(out_dir)
#         self.pricing_model = joblib.load(p/"pricing_model.pkl")
#         self.scaler = joblib.load(p/"pricing_scaler.pkl")
#         self.pricing_rules = json.load(open(p/"pricing_rules.json"))
#         self.market_factors = json.load(open(p/"market_factors.json"))
#         meta = json.load(open(p/"meta.json"))
#         self.base_premium = float(meta.get("base_premium", self.base_premium))
#         self._ml_train_cols = meta.get("train_cols", [])


# # -----------------------------
# # CLI Demo
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--features", type=str, default="data/features.csv")
#     ap.add_argument("--trip_scores", type=str, default="data/trip_scores.csv",
#                     help="Optional: contains p_claim, expected_severity, expected_loss, risk_score_trip")
#     ap.add_argument("--out_dir", type=str, default="models_pricing")
#     ap.add_argument("--base_premium", type=float, default=1200.0)
#     args = ap.parse_args()

#     feats, scores = load_inputs(args.features, args.trip_scores)

#     # Use driver-level features from your pipeline. Minimal required cols:
#     needed = [
#         "miles","duration_sec","over_limit","harsh_brakes","night","rain",
#         "crime_index","accident_index","repair_cost_index",
#         "vehicle_age","safety_score",
#         "prior_claims_count","tickets_count","dui_flag",
#         "age_18_24","age_25_34","age_35_49","age_50_64","age_65_plus",
#         "claim","severity"
#     ]
#     feats = ensure_columns(feats, needed)

#     # Risk score & expected loss input
#     if scores is not None:
#         # join on trip_id if present, else row-wise align
#         join_key = "trip_id" if "trip_id" in feats.columns and "trip_id" in scores.columns else None
#         if join_key:
#             df = feats.merge(scores[["trip_id","expected_loss","risk_score_trip"]], on="trip_id", how="left")
#         else:
#             df = feats.copy()
#             for c in ["expected_loss","risk_score_trip"]:
#                 if c in scores.columns:
#                     df[c] = scores[c].values[:len(df)]
#     else:
#         df = feats.copy()
#         # fallback risk score proxy (simple)
#         df["risk_score_trip"] = (
#             100
#             - 40*df["over_limit"].clip(0,1)
#             - 35*df["harsh_brakes"].clip(0,1)
#             - 15*df["night"].clip(0,1)
#             - 10*df["rain"].clip(0,1)
#             + 8*np.clip((df["safety_score"]-5)/5, -1, 1)
#         ).clip(0,100)
#         # fallback expected loss proxy
#         df["expected_loss"] = (0.08 + 0.6*df["over_limit"] + 0.5*df["harsh_brakes"]) \
#                               * (1500 + 2500*df["over_limit"] + 2000*df["harsh_brakes"])

#     risk = df["risk_score_trip"].fillna(50).to_numpy()
#     exp_loss = df["expected_loss"].fillna(df["expected_loss"].median()).to_numpy()

#     # 1) Rule-based pricing
#     engine = TelematicsPricingEngine(base_premium=args.base_premium)
#     premiums_rule, details = engine.price_rule_based(df, risk)

#     # Optional: calibrate premiums to hit portfolio loss ratio target
#     premiums_rule_cal = engine.calibrate_to_loss_ratio(premiums_rule, exp_loss)

#     # 2) ML pricing (targets default to calibrated rule premiums)
#     print("\n[ML] Training pricing model …")
#     ml_metrics = engine.train_ml_pricer(df, risk, target_premiums=premiums_rule_cal)
#     premiums_ml = engine.predict_ml(df, risk)

#     # 3) Analysis & visuals
#     analysis_df, summary = engine.analyze(
#         df, risk, premiums_rule_cal, premiums_ml,
#         actual_claim=feats.get("claim", None), severity=feats.get("severity", None)
#     )

#     print("\n[SUMMARY] Risk–Premium correlations:")
# #     print(f"  Rule-based: {summary['metrics']['corr_rule']:.4f}")
# #     print(f"  ML-based  : {summary['metrics']['corr_ml']:.4f}\n")
# #     print("[Segments] Averages:\n", summary["segments"].to_string(index=False))

# #     engine.plot_all(analysis_df, base_premium=args.base_premium)

# #     # 4) Persist artifacts & exports
# #     out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
# #     engine.save(args.out_dir)
# #     analysis_df.to_csv(out/"pricing_analysis.csv", index=False)
# #     pd.DataFrame(details).to_csv(out/"rule_breakdown_sample.csv", index=False)
# #     print(f"\n[DONE] Wrote analysis to {out/'pricing_analysis.csv'}")

# # if __name__ == "__main__":
# #     main()
# #!/usr/bin/env python3
# """
# Pricing Engine for Telematics Claim Risk (consumes trained artifacts)
# ----------------------------------------------------------------------
# This script loads the saved artifacts produced by the training pipeline
# (preprocessor.joblib, claim_calibrated.joblib, meta.json and optionally
# severity_best.joblib) and generates:

# - Trip-level pricing recommendations
# - Driver- and Vehicle-level aggregated pricing (exposure-weighted by miles)

# It makes **no changes** to model weights; it only scores and prices.

# Design notes
# - Uses the same leakage policy as training: drops IDs and post-claim fields
# - Uses the calibrated classifier to compute p_claim
# - Expected Severity sources (hierarchy):
#   1) If an explicit severity model artifact exists, use it (and inverse the log1p if trained that way)
#   2) Else, if input has historical `severity` for rows where claim==1, use the positive-claim mean as a fallback
#   3) Else, use a fixed `severity_fallback` from config/CLI
# - Pricing tiers are **configurable**; defaults are provided below

# Outputs (in --outdir)
# - trip_pricing.csv
# - driver_pricing.csv
# - vehicle_pricing.csv
# - pricing_meta.json

# Usage
#   python pricing_engine.py \
#     --features /path/to/score.csv \
#     --artifacts_dir /path/to/models \
#     --outdir /path/to/out \
#     --base_premium 1000
# """

# import argparse, json, logging
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import joblib

# from sklearn.preprocessing import FunctionTransformer

# # --------------------
# # Logging
# # --------------------
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# L = logging.getLogger("pricing_engine")

# # --------------------
# # Columns & leakage policy (must mirror training)
# # --------------------
# TARGET = "claim"
# ID_COLS = ["trip_id","user_id","vehicle_id","segment_key"]
# POST_CLAIM_SUSPECTS = ["severity","repair_cost_index","parts_availability_index"]

# # --------------------
# # Default pricing configuration
# # --------------------
# DEFAULT_TIERS = [
#     {"name": "Very Low",  "prob_lo": 0.00, "prob_hi": 0.10, "mult": 0.85},
#     {"name": "Low",       "prob_lo": 0.10, "prob_hi": 0.20, "mult": 0.95},
#     {"name": "Medium",    "prob_lo": 0.20, "prob_hi": 0.40, "mult": 1.05},
#     {"name": "High",      "prob_lo": 0.40, "prob_hi": 0.70, "mult": 1.20},
#     {"name": "Very High", "prob_lo": 0.70, "prob_hi": 1.01, "mult": 1.40},
# ]

# # Caps and guards (actuarial/ops constraints)
# MIN_MULT = 0.70
# MAX_MULT = 1.75
# MAX_INDIVIDUAL_DELTA = 0.35  # maximum single-step change from current premium if provided


# # --------------------
# # Helpers
# # --------------------

# def load_artifacts(Path):
#     pre = joblib.load("models/preprocessor.joblib")
#     clf_cal = joblib.load("models/claim_calibrated.joblib")
#     meta = json.loads(("models/meta.json").read_text())

#     # Optional severity model
#     sev_model = None
#     sev_path = "models/severity_best.joblib"
#     if sev_path.exists():
#         try:
#             sev_model = joblib.load(sev_path)
#             L.info("Loaded severity model artifact.")
#         except Exception as e:
#             L.warning(f"Severity artifact present but failed to load: {e}")
#             sev_model = None

#     return pre, clf_cal, meta, sev_model


# def drop_leakage(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
#     removed: List[str] = []
#     for c in POST_CLAIM_SUSPECTS + ID_COLS:
#         if c in df.columns:
#             removed.append(c)
#     if removed:
#         df = df.drop(columns=removed)
#     return df, removed


# def choose_tier(p: float, tiers: List[Dict]) -> Tuple[str, float]:
#     for t in tiers:
#         if t["prob_lo"] <= p < t["prob_hi"]:
#             return t["name"], float(t["mult"])
#     return tiers[-1]["name"], float(tiers[-1]["mult"])  # fallback


# def cap_multiplier(mult: float) -> float:
#     return float(np.clip(mult, MIN_MULT, MAX_MULT))


# # --------------------
# # Severity estimation
# # --------------------

# def estimate_severity(
#     df_raw: pd.DataFrame,
#     X_processed: np.ndarray,
#     sev_model,
#     severity_fallback: float
# ) -> np.ndarray:
#     """Return expected severity for each row to support expected loss.
#     If a severity model exists, use it; otherwise try empirical mean severity
#     (when historical severity is present for positive claims), else fallback.
#     """
#     if sev_model is not None:
#         try:
#             sev_log = sev_model.predict(X_processed)
#             sev = np.expm1(sev_log).clip(min=0)
#             return sev.astype(float)
#         except Exception as e:
#             L.warning(f"Severity model predict failed, falling back: {e}")

#     # If historical severity exists, compute mean over positive claims
#     if "severity" in df_raw.columns and "claim" in df_raw.columns:
#         pos = (pd.to_numeric(df_raw["claim"], errors="coerce") == 1) & (
#             pd.to_numeric(df_raw["severity"], errors="coerce").fillna(0) > 0
#         )
#         if pos.any():
#             m = float(pd.to_numeric(df_raw.loc[pos, "severity"], errors="coerce").fillna(0).mean())
#             if np.isfinite(m) and m > 0:
#                 return np.full(len(df_raw), m, dtype=float)

#     return np.full(len(df_raw), float(severity_fallback), dtype=float)


# # --------------------
# # Engine
# # --------------------

# class PricingEngine:
#     def __init__(self, artifacts_dir: str, tiers: Optional[List[Dict]] = None,
#                  base_premium: float = 1000.0, severity_fallback: float = 2500.0):
#         self.artifacts_dir = Path(artifacts_dir)
#         self.tiers = tiers or DEFAULT_TIERS
#         self.base_premium = float(base_premium)
#         self.severity_fallback = float(severity_fallback)

#         self.pre, self.clf_cal, self.meta, self.sev_model = load_artifacts(self.artifacts_dir)
#         self.removed_at_train = set(self.meta.get("removed_columns", []))

#     def score(self, df: pd.DataFrame) -> pd.DataFrame:
#         df_in = df.copy()
#         # Keep original IDs for output if present
#         ids = {c: df_in[c] for c in ID_COLS if c in df_in.columns}

#         # Align columns: drop leakage/IDs now
#         df_sc, removed = drop_leakage(df_in)

#         # Ensure training columns exist; add missing with zeros
#         # (OneHotEncoder handles cats, so we rely on preprocessor to align)
#         X = self.pre.transform(df_sc.drop(columns=[TARGET]) if TARGET in df_sc.columns else df_sc)

#         p_claim = self.clf_cal.predict_proba(X)[:, 1]
#         sev = estimate_severity(df_in, X, self.sev_model, self.severity_fallback)
#         expected_loss = p_claim * sev

#         # Tier & multiplier
#         tier_names, mults = [], []
#         for p in p_claim:
#             name, m = choose_tier(float(p), self.tiers)
#             tier_names.append(name)
#             mults.append(cap_multiplier(m))

#         out = pd.DataFrame({
#             **({k: v for k, v in ids.items()}),
#             "p_claim": p_claim,
#             "expected_severity": sev,
#             "expected_loss": expected_loss,
#             "tier": tier_names,
#             "premium_multiplier": mults,
#             "suggested_premium": np.array(mults) * self.base_premium,
#         })
#         return out

#     @staticmethod
#     def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
#         w = np.asarray(weights, dtype=float)
#         x = np.asarray(series, dtype=float)
#         w = np.where(np.isfinite(w) & (w > 0), w, 1e-6)
#         return float(np.average(x, weights=w))

#     def aggregate(self, trip_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         # Exposure weights by miles when available
#         miles = pd.to_numeric(trip_df.get("miles", pd.Series(np.ones(len(trip_df)))), errors="coerce").fillna(1.0)

#         # Driver level
#         if "user_id" in trip_df.columns:
#             g = trip_df.groupby("user_id", dropna=False)
#             driver = g.apply(lambda t: pd.Series({
#                 "expected_loss": self._weighted_mean(t["expected_loss"], miles.loc[t.index]),
#                 "premium_multiplier": self._weighted_mean(t["premium_multiplier"], miles.loc[t.index]),
#                 "suggested_premium": self._weighted_mean(t["suggested_premium"], miles.loc[t.index]),
#                 "miles": miles.loc[t.index].sum(),
#             })).reset_index()
#         else:
#             driver = pd.DataFrame(columns=["user_id","expected_loss","premium_multiplier","suggested_premium","miles"])  # empty

#         # Vehicle level
#         if "vehicle_id" in trip_df.columns:
#             gv = trip_df.groupby("vehicle_id", dropna=False)
#             vehicle = gv.apply(lambda t: pd.Series({
#                 "expected_loss": self._weighted_mean(t["expected_loss"], miles.loc[t.index]),
#                 "premium_multiplier": self._weighted_mean(t["premium_multiplier"], miles.loc[t.index]),
#                 "suggested_premium": self._weighted_mean(t["suggested_premium"], miles.loc[t.index]),
#                 "miles": miles.loc[t.index].sum(),
#             })).reset_index()
#         else:
#             vehicle = pd.DataFrame(columns=["vehicle_id","expected_loss","premium_multiplier","suggested_premium","miles"])  # empty

#         return driver, vehicle


# # --------------------
# # CLI
# # --------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--features", required=True, help="CSV to score for pricing")
#     ap.add_argument("--artifacts_dir", required=True, help="Directory with training artifacts")
#     ap.add_argument("--outdir", default="./pricing_out", help="Where to write outputs")
#     ap.add_argument("--base_premium", type=float, default=1000.0)
#     ap.add_argument("--severity_fallback", type=float, default=2500.0)
#     ap.add_argument("--tiers_json", type=str, default=None, help="Optional JSON file with pricing tiers")
#     args = ap.parse_args()

#     outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

#     # Load optional tiers
#     tiers = None
#     if args.tiers_json:
#         try:
#             tiers = json.loads(Path(args.tiers_json).read_text())
#             assert isinstance(tiers, list) and len(tiers) > 0
#             L.info("Loaded custom pricing tiers from JSON.")
#         except Exception as e:
#             L.warning(f"Failed to load custom tiers, using defaults: {e}")
#             tiers = None

#     engine = PricingEngine(
#         artifacts_dir=args.artifacts_dir,
#         tiers=tiers,
#         base_premium=args.base_premium,
#         severity_fallback=args.severity_fallback,
#     )

#     df = pd.read_csv(args.features)
#     trips = engine.score(df)
#     drivers, vehicles = engine.aggregate(trips)

#     # Save
#     trips.to_csv(outdir / "trip_pricing.csv", index=False)
#     drivers.to_csv(outdir / "driver_pricing.csv", index=False)
#     vehicles.to_csv(outdir / "vehicle_pricing.csv", index=False)

#     meta = {
#         "artifacts_dir": str(Path(args.artifacts_dir).resolve()),
#         "base_premium": float(args.base_premium),
#         "severity_fallback": float(args.severity_fallback),
#         "tiers": engine.tiers,
#         "caps": {"min_mult": MIN_MULT, "max_mult": MAX_MULT, "max_individual_delta": MAX_INDIVIDUAL_DELTA},
#     }
#     (outdir / "pricing_meta.json").write_text(json.dumps(meta, indent=2))

#     L.info(f"Saved pricing outputs -> {outdir}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Production Insurance Pricing Engine

Clean, professional implementation for behavioral risk-based pricing.
Integrates ML risk scores with traditional actuarial factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PricingConfig:
    """Pricing configuration with industry-standard parameters"""
    
    # Base pricing
    base_annual_premium: float = 1200.0
    min_premium: float = 600.0
    max_premium: float = 3500.0
    
    # Risk model weight (0-1, rest is traditional factors)
    behavioral_weight: float = 0.4
    
    # Age factor table (industry standard)
    age_factors: Dict[str, float] = field(default_factory=lambda: {
        '16-20': 1.75, '21-25': 1.35, '26-35': 1.00, 
        '36-50': 0.90, '51-65': 0.85, '66+': 1.10
    })
    
    # Experience factor
    experience_factors: Dict[str, float] = field(default_factory=lambda: {
        'new': 1.25, 'intermediate': 1.00, 'experienced': 0.95
    })
    
    # Vehicle age factor
    vehicle_age_factors: Dict[str, float] = field(default_factory=lambda: {
        '0-2': 1.20, '3-5': 1.10, '6-10': 1.00, '11-15': 0.95, '16+': 0.90
    })
    
    # Coverage adjustments
    coverage_factors: Dict[str, float] = field(default_factory=lambda: {
        'liability_only': 0.75, 'standard': 1.00, 'comprehensive': 1.25
    })

class InsurancePricingEngine:
    """
    Professional insurance pricing engine with ML integration
    """
    
    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()
        self.pricing_history = []
    
    def categorize_age(self, age: int) -> str:
        """Map age to pricing category"""
        if age <= 20: return '16-20'
        elif age <= 25: return '21-25'
        elif age <= 35: return '26-35'
        elif age <= 50: return '36-50'
        elif age <= 65: return '51-65'
        else: return '66+'
    
    def categorize_experience(self, years_licensed: int) -> str:
        """Map driving experience to category"""
        if years_licensed < 3: return 'new'
        elif years_licensed < 10: return 'intermediate'
        else: return 'experienced'
    
    def categorize_vehicle_age(self, vehicle_year: int) -> str:
        """Map vehicle year to age category"""
        current_year = datetime.now().year
        age = current_year - vehicle_year
        
        if age <= 2: return '0-2'
        elif age <= 5: return '3-5'
        elif age <= 10: return '6-10'
        elif age <= 15: return '11-15'
        else: return '16+'
    
    def calculate_behavioral_factor(self, risk_score: float) -> float:
        """
        Convert ML risk score (0-100) to pricing multiplier
        
        Args:
            risk_score: Behavioral risk score from ML model
            
        Returns:
            Multiplier between 0.7 and 1.5
        """
        # Normalize to 0-1, then map to multiplier range
        normalized = np.clip(risk_score / 100.0, 0, 1)
        return 0.7 + (normalized * 0.8)  # Range: 0.7x to 1.5x
    
    def calculate_traditional_factors(self, driver_profile: Dict) -> float:
        """
        Calculate traditional actuarial factors
        
        Args:
            driver_profile: Driver and vehicle information
            
        Returns:
            Combined traditional factor multiplier
        """
        factor = 1.0
        
        # Age factor
        age = driver_profile.get('age', 35)
        age_category = self.categorize_age(age)
        factor *= self.config.age_factors.get(age_category, 1.0)
        
        # Experience factor
        experience = driver_profile.get('years_licensed', 10)
        exp_category = self.categorize_experience(experience)
        factor *= self.config.experience_factors.get(exp_category, 1.0)
        
        # Vehicle age factor
        vehicle_year = driver_profile.get('vehicle_year', 2015)
        vehicle_category = self.categorize_vehicle_age(vehicle_year)
        factor *= self.config.vehicle_age_factors.get(vehicle_category, 1.0)
        
        # Coverage level
        coverage = driver_profile.get('coverage_level', 'standard')
        factor *= self.config.coverage_factors.get(coverage, 1.0)
        
        # Claims history penalty
        prior_claims = driver_profile.get('prior_claims', 0)
        if prior_claims > 0:
            factor *= (1 + prior_claims * 0.15)  # 15% penalty per claim
        
        # DUI penalty
        if driver_profile.get('dui_flag', False):
            factor *= 1.50
        
        return factor
    
    def apply_business_rules(self, premium: float, driver_profile: Dict) -> float:
        """
        Apply final business rules and constraints
        
        Args:
            premium: Calculated premium before business rules
            driver_profile: Driver information
            
        Returns:
            Final premium after business rules
        """
        # Mileage adjustment
        annual_mileage = driver_profile.get('annual_mileage', 12000)
        if annual_mileage > 25000:
            premium *= 1.15
        elif annual_mileage < 5000:
            premium *= 0.95
        
        # Multi-vehicle discount
        if driver_profile.get('num_vehicles', 1) > 1:
            premium *= 0.95
        
        # Good student discount
        if driver_profile.get('good_student', False) and driver_profile.get('age', 30) < 25:
            premium *= 0.90
        
        # Ensure premium stays within regulatory bounds
        premium = max(self.config.min_premium, 
                     min(premium, self.config.max_premium))
        
        return premium
    
    def calculate_premium(self, risk_score: float, driver_profile: Dict) -> Dict:
        """
        Main pricing function combining ML and traditional factors
        
        Args:
            risk_score: ML behavioral risk score (0-100)
            driver_profile: Driver and vehicle characteristics
            
        Returns:
            Complete pricing breakdown
        """
        logger.info(f"Calculating premium for risk score: {risk_score:.1f}")
        
        # Calculate component factors
        behavioral_factor = self.calculate_behavioral_factor(risk_score)
        traditional_factor = self.calculate_traditional_factors(driver_profile)
        
        # Weighted combination
        combined_factor = (
            self.config.behavioral_weight * behavioral_factor +
            (1 - self.config.behavioral_weight) * traditional_factor
        )
        
        # Base calculation
        base_premium = self.config.base_annual_premium * combined_factor
        
        # Apply business rules
        final_premium = self.apply_business_rules(base_premium, driver_profile)
        
        # Determine tier
        tier = self._determine_tier(risk_score, final_premium)
        
        # Create comprehensive result
        result = {
            'annual_premium': round(final_premium, 2),
            'monthly_premium': round(final_premium / 12, 2),
            'risk_score': risk_score,
            'pricing_tier': tier,
            'factors': {
                'behavioral_factor': round(behavioral_factor, 3),
                'traditional_factor': round(traditional_factor, 3),
                'combined_factor': round(combined_factor, 3)
            },
            'components': {
                'base_premium': self.config.base_annual_premium,
                'behavioral_component': round(self.config.base_annual_premium * behavioral_factor * self.config.behavioral_weight, 2),
                'traditional_component': round(self.config.base_annual_premium * traditional_factor * (1 - self.config.behavioral_weight), 2)
            },
            'profile_summary': {
                'age_category': self.categorize_age(driver_profile.get('age', 35)),
                'experience_level': self.categorize_experience(driver_profile.get('years_licensed', 10)),
                'vehicle_age_category': self.categorize_vehicle_age(driver_profile.get('vehicle_year', 2015))
            },
            'calculated_at': datetime.now().isoformat()
        }
        
        # Store for analytics
        self.pricing_history.append({
            'risk_score': risk_score,
            'premium': final_premium,
            'tier': tier,
            'timestamp': datetime.now()
        })
        
        return result
    
    def _determine_tier(self, risk_score: float, premium: float) -> str:
        """Determine pricing tier for marketing purposes"""
        base_ratio = premium / self.config.base_annual_premium
        
        if risk_score < 25 and base_ratio < 0.9:
            return 'Preferred Plus'
        elif risk_score < 50 and base_ratio < 1.1:
            return 'Preferred'
        elif risk_score < 75 and base_ratio < 1.3:
            return 'Standard'
        else:
            return 'Non-Standard'
    
    def batch_pricing(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        """
        Price multiple policies efficiently
        
        Args:
            pricing_data: DataFrame with risk scores and driver profiles
            
        Returns:
            DataFrame with pricing results
        """
        logger.info(f"Processing batch pricing for {len(pricing_data)} policies")
        
        results = []
        
        for _, row in pricing_data.iterrows():
            # Extract risk score
            risk_score = row.get('risk_score', 50.0)
            
            # Build driver profile from row data
            driver_profile = {
                'age': row.get('age', 35),
                'years_licensed': row.get('years_licensed', 10),
                'vehicle_year': row.get('vehicle_year', 2015),
                'coverage_level': row.get('coverage_level', 'standard'),
                'prior_claims': row.get('prior_claims', 0),
                'dui_flag': row.get('dui_flag', False),
                'annual_mileage': row.get('annual_mileage', 12000),
                'num_vehicles': row.get('num_vehicles', 1),
                'good_student': row.get('good_student', False)
            }
            
            # Calculate pricing
            pricing_result = self.calculate_premium(risk_score, driver_profile)
            
            # Extract key results
            result_row = {
                'driver_id': row.get('driver_id', f"driver_{len(results)}"),
                'risk_score': risk_score,
                'annual_premium': pricing_result['annual_premium'],
                'monthly_premium': pricing_result['monthly_premium'],
                'pricing_tier': pricing_result['pricing_tier'],
                'behavioral_factor': pricing_result['factors']['behavioral_factor'],
                'traditional_factor': pricing_result['factors']['traditional_factor']
            }
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def get_pricing_analytics(self) -> Dict:
        """Generate pricing analytics from history"""
        if not self.pricing_history:
            return {}
        
        df = pd.DataFrame(self.pricing_history)
        
        return {
            'total_quotes': len(df),
            'average_premium': df['premium'].mean(),
            'premium_range': (df['premium'].min(), df['premium'].max()),
            'tier_distribution': df['tier'].value_counts().to_dict(),
            'risk_score_stats': {
                'mean': df['risk_score'].mean(),
                'std': df['risk_score'].std(),
                'quartiles': df['risk_score'].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        }

class PricingIntegration:
    """Integration layer for ML models and pricing engine"""
    
    def __init__(self, risk_scorer, pricing_engine: InsurancePricingEngine):
        self.risk_scorer = risk_scorer
        self.pricing_engine = pricing_engine
    
    def quote_driver(self, driver_features: pd.DataFrame, driver_profile: Dict) -> Dict:
        """
        Complete workflow: risk scoring + pricing
        
        Args:
            driver_features: Processed features for ML model
            driver_profile: Traditional rating factors
            
        Returns:
            Complete quote with risk assessment and pricing
        """
        # Get risk score from ML model
        risk_result = self.risk_scorer.calculate_risk_score(driver_features)
        risk_score = risk_result['risk_score'].iloc[0]
        
        # Calculate pricing
        pricing_result = self.pricing_engine.calculate_premium(risk_score, driver_profile)
        
        # Combine into complete quote
        quote = {
            **pricing_result,
            'risk_assessment': {
                'risk_score': float(risk_score),
                'risk_category': risk_result['risk_category'].iloc[0],
                'risk_probability': float(risk_result['risk_probability'].iloc[0])
            },
            'quote_id': f"Q{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        return quote

# Example usage and demo
def demo_pricing_engine():
    """Demonstrate pricing engine capabilities"""
    
    print("Insurance Pricing Engine Demo")
    print("=" * 50)
    
    # Initialize engine
    config = PricingConfig(
        base_annual_premium=1200,
        behavioral_weight=0.4
    )
    engine = InsurancePricingEngine(config)
    
    # Example drivers with different risk profiles
    test_drivers = [
        {
            'name': 'Low Risk Driver',
            'risk_score': 25,
            'profile': {
                'age': 35, 'years_licensed': 15, 'vehicle_year': 2020,
                'prior_claims': 0, 'dui_flag': False, 'annual_mileage': 10000
            }
        },
        {
            'name': 'High Risk Driver', 
            'risk_score': 85,
            'profile': {
                'age': 22, 'years_licensed': 2, 'vehicle_year': 2005,
                'prior_claims': 2, 'dui_flag': True, 'annual_mileage': 20000
            }
        }
    ]
    
    for driver in test_drivers:
        print(f"\n{driver['name']}:")
        result = engine.calculate_premium(driver['risk_score'], driver['profile'])
        
        print(f"  Risk Score: {result['risk_score']}")
        print(f"  Annual Premium: ${result['annual_premium']:,.2f}")
        print(f"  Monthly Premium: ${result['monthly_premium']:,.2f}")
        print(f"  Pricing Tier: {result['pricing_tier']}")
        print(f"  Behavioral Factor: {result['factors']['behavioral_factor']:.3f}")
        print(f"  Traditional Factor: {result['factors']['traditional_factor']:.3f}")

if __name__ == "__main__":
    demo_pricing_engine()