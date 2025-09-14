# src/backend/price_engine_run.py
"""
Pricing Engine Runner

Script to run the pricing engine on your risk scores and driver data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import your modules
from pricing_engine import InsurancePricingEngine, PricingConfig, PricingIntegration
from risk_scoring import RiskScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_pricing_data(risk_scores_df: pd.DataFrame, driver_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine risk scores with driver profile data for pricing
    
    Args:
        risk_scores_df: Output from your risk scoring (behavioral_risk_scores.csv)
        driver_data: Driver profile data with traditional rating factors
        
    Returns:
        Combined DataFrame ready for pricing
    """
    logger.info("Preparing pricing data...")
    
    # Merge risk scores with driver data
    if 'user_id' in risk_scores_df.columns and 'user_id' in driver_data.columns:
        pricing_df = risk_scores_df.merge(driver_data, on='user_id', how='left')
    else:
        # If no user_id, assume same order
        pricing_df = pd.concat([risk_scores_df.reset_index(drop=True), 
                               driver_data.reset_index(drop=True)], axis=1)
    
    # Ensure required columns exist with defaults
    required_cols = {
        'risk_score': 50.0,
        'age': 35,
        'years_licensed': 10,
        'vehicle_year': 2015,
        'coverage_level': 'standard',
        'prior_claims': 0,
        'dui_flag': False,
        'annual_mileage': 12000,
        'num_vehicles': 1,
        'good_student': False}
    
    for col, default_val in required_cols.items():
        if col not in pricing_df.columns:
            pricing_df[col] = default_val
            logger.info(f"Added missing column '{col}' with default value: {default_val}")
    
    # Clean data
    pricing_df['dui_flag'] = pricing_df.get('dui_flag', 0).astype(bool)
    pricing_df['good_student'] = pricing_df.get('good_student', 0).astype(bool)
    pricing_df['prior_claims'] = pricing_df.get('prior_claims_count', pricing_df.get('prior_claims', 0))
    
    # Map existing columns if they have different names
    column_mapping = {
        'annual_mileage_declared': 'annual_mileage',
        'prior_claims_count': 'prior_claims',
        'year': 'vehicle_year'  # If vehicle year is in 'year' column
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in pricing_df.columns and new_name not in pricing_df.columns:
            pricing_df[new_name] = pricing_df[old_name]
    
    logger.info(f"Prepared pricing data for {len(pricing_df)} drivers")
    return pricing_df

def run_pricing_engine(data_path: str = "./"):
    """
    Run pricing engine on your data
    
    Args:
        data_path: Path to directory containing your data files
    """
    
    print("=" * 80)
    print("INSURANCE PRICING ENGINE")
    print("=" * 80)
    
    data_path = Path(data_path)
    
    try:
        # ================================================================
        # STEP 1: LOAD DATA
        # ================================================================
        print("\nSTEP 1: Loading Data...")
        print("-" * 50)
        
        # Load risk scores (from your risk scoring output)
        risk_scores_file = data_path / "data/risk_socres/behavioral_risk_scores.csv"
        if risk_scores_file.exists():
            risk_scores_df = pd.read_csv(risk_scores_file)
            print(f"✓ Loaded risk scores: {risk_scores_df.shape}")
        else:
            print(f"Risk scores file not found: {risk_scores_file}")
            print("Run risk_scoring_run.py first to generate risk scores")
            return
        
        # Load driver profile data
        driver_profile_file = data_path / "data/driver_profile.csv"
        if driver_profile_file.exists():
            driver_data = pd.read_csv(driver_profile_file)
            print(f"✓ Loaded driver profiles: {driver_data.shape}")
        else:
            print("Driver profile file not found, using risk score data only")
            driver_data = pd.DataFrame()
        
        # Load vehicle info if available
        vehicle_file = data_path / "data/vehicle_info.csv"
        vehicle_assignments_file = data_path / "data/vehicle_assignments.csv"
        
        if vehicle_file.exists() and vehicle_assignments_file.exists():
            vehicle_info = pd.read_csv(vehicle_file)
            vehicle_assignments = pd.read_csv(vehicle_assignments_file)
            
            # Merge vehicle data
            vehicle_data = vehicle_assignments.merge(vehicle_info, on='vehicle_id', how='left')
            
            if not driver_data.empty:
                driver_data = driver_data.merge(vehicle_data, on='user_id', how='left')
            else:
                driver_data = vehicle_data
            
            print(f"✓ Loaded vehicle data: {vehicle_info.shape}")
        
        # ================================================================
        # STEP 2: PREPARE PRICING DATA
        # ================================================================
        print(f"\nSTEP 2: Preparing Pricing Data...")
        print("-" * 50)
        
        pricing_df = prepare_pricing_data(risk_scores_df, driver_data)
        
        print(f"Data columns available: {list(pricing_df.columns)}")
        print(f"Sample risk scores: {pricing_df['risk_score'].describe()}")
        
        # ================================================================
        # STEP 3: CONFIGURE PRICING ENGINE
        # ================================================================
        print(f"\nSTEP 3: Configuring Pricing Engine...")
        print("-" * 50)
        
        # Configure pricing engine
        config = PricingConfig(
            base_annual_premium=1200.0,     # Adjust for your market
            behavioral_weight=0.4,          # 40% ML, 60% traditional factors
            min_premium=600.0,
            max_premium=3500.0
        )
        
        engine = InsurancePricingEngine(config)
        print(f"✓ Pricing engine configured")
        print(f"  Base premium: ${config.base_annual_premium}")
        print(f"  Behavioral weight: {config.behavioral_weight*100}%")
        
        # ================================================================
        # STEP 4: CALCULATE PREMIUMS
        # ================================================================
        print(f"\nSTEP 4: Calculating Premiums...")
        print("-" * 50)
        
        # Use batch pricing for efficiency
        pricing_results = engine.batch_pricing(pricing_df)
        
        print(f"✓ Calculated premiums for {len(pricing_results)} drivers")
        
        # ================================================================
        # STEP 5: ANALYZE RESULTS
        # ================================================================
        print(f"\nSTEP 5: Analyzing Results...")
        print("-" * 50)
        
        # Premium statistics
        print(f"Premium Statistics:")
        print(f"  Average: ${pricing_results['annual_premium'].mean():,.2f}")
        print(f"  Median:  ${pricing_results['annual_premium'].median():,.2f}")
        print(f"  Range:   ${pricing_results['annual_premium'].min():,.2f} - ${pricing_results['annual_premium'].max():,.2f}")
        
        # Tier distribution
        print(f"\nPricing Tier Distribution:")
        tier_dist = pricing_results['pricing_tier'].value_counts()
        for tier, count in tier_dist.items():
            pct = count / len(pricing_results) * 100
            print(f"  {tier}: {count} ({pct:.1f}%)")
        
        # Risk score vs premium correlation
        correlation = pricing_results['risk_score'].corr(pricing_results['annual_premium'])
        print(f"\nRisk Score vs Premium Correlation: {correlation:.3f}")
        
        # ================================================================
        # STEP 6: SAVE RESULTS
        # ================================================================
        print(f"\nSTEP 6: Saving Results...")
        print("-" * 50)
        
        # Save pricing results
        output_file = data_path / "data/insurance_quotes.csv"
        pricing_results.to_csv(output_file, index=False)
        print(f"✓ Pricing results saved to: {output_file}")
        
        # Save detailed breakdown for sample drivers
        detailed_output = data_path / "data/detailed_quotes_sample.csv"
        sample_detailed = []
        
        for i in range(min(10, len(pricing_df))):
            row = pricing_df.iloc[i]
            
            # Get detailed quote
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
            
            detailed_quote = engine.calculate_premium(row['risk_score'], driver_profile)
            
            # Flatten for CSV
            flat_quote = {
                'driver_id': row.get('user_id', f'driver_{i}'),
                'risk_score': detailed_quote['risk_score'],
                'annual_premium': detailed_quote['annual_premium'],
                'monthly_premium': detailed_quote['monthly_premium'],
                'pricing_tier': detailed_quote['pricing_tier'],
                'behavioral_factor': detailed_quote['factors']['behavioral_factor'],
                'traditional_factor': detailed_quote['factors']['traditional_factor'],
                'base_premium': detailed_quote['components']['base_premium'],
                'behavioral_component': detailed_quote['components']['behavioral_component'],
                'traditional_component': detailed_quote['components']['traditional_component']}
            
            sample_detailed.append(flat_quote)
        
        pd.DataFrame(sample_detailed).to_csv(detailed_output, index=False)
        print(f"✓ Detailed quotes sample saved to: {detailed_output}")
        
        # ================================================================
        # STEP 7: SAMPLE QUOTES
        # ================================================================
        print(f"\nSTEP 7: Sample Quotes...")
        print("-" * 50)
        
        # Show sample quotes
        print("Sample Insurance Quotes:")
        display_cols = ['driver_id', 'risk_score', 'annual_premium', 'monthly_premium', 'pricing_tier']
        print(pricing_results[display_cols].head(10).to_string(index=False))
        
        # Show highest and lowest premiums
        print(f"\nHighest Premiums:")
        top_premiums = pricing_results.nlargest(5, 'annual_premium')[display_cols]
        print(top_premiums.to_string(index=False))
        
        print(f"\nLowest Premiums:")
        low_premiums = pricing_results.nsmallest(5, 'annual_premium')[display_cols]
        print(low_premiums.to_string(index=False))
        
        # ================================================================
        # COMPLETION
        # ================================================================
        print(f"\n" + "="*80)
        print("PRICING ENGINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"{len(pricing_results)} quotes generated")
        print(f"Average premium: ${pricing_results['annual_premium'].mean():,.2f}")
        print(f"Results saved to: insurance_quotes.csv")
        
        return pricing_results
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting Insurance Pricing Engine...")
    
    # Update this path to your data directory
    DATA_PATH = "./"
    
    # Run pricing engine
    results = run_pricing_engine(DATA_PATH)
