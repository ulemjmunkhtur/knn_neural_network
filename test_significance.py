import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Function to perform ANOVA and determine significance
def perform_anova(data, dependent_var):
    """
    Perform ANOVA on the given data to determine the significance of factors.

    Parameters:
    data (pd.DataFrame): The data containing the experimental results.
    dependent_var (str): The dependent variable for the ANOVA.

    Returns:
    pd.DataFrame: The ANOVA table with significance.
    statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted model.
    """
    formula = f'{dependent_var} ~ C(k_ratio) + C(initialization_method) + C(use_samples) + C(use_mixup) + ' \
              f'C(k_ratio):C(initialization_method) + C(k_ratio):C(use_samples) + C(k_ratio):C(use_mixup) + ' \
              f'C(initialization_method):C(use_samples) + C(initialization_method):C(use_mixup) + ' \
              f'C(use_samples):C(use_mixup)'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Mark significance
    anova_table['Significant'] = anova_table['PR(>F)'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
    return anova_table, model


def analyze_and_output(csv_path, output_path):
    """
    Read data from a CSV file, perform ANOVA and regression analysis, and save the results to an Excel file.

    Parameters:
    csv_path (str): Path to the input CSV file.
    output_path (str): Path to the output Excel file.
    """

    data = pd.read_csv(csv_path)

    dependent_vars = ['good_convergence_rate', 'doesnt_converge_rate', 'collapse_rate', 'average_iterations', 'average_jaccard_score']

    # Perform ANOVA and store results
    anova_results = {}
    model_results = {}
    for var in dependent_vars:
        anova_table, model = perform_anova(data, var)
        anova_results[var] = anova_table
        model_results[var] = model

    # Write results to output Excel
    with pd.ExcelWriter(output_path) as writer:
        for var, result in anova_results.items():
            result.to_excel(writer, sheet_name=f'{var}_anova')

        for var, model in model_results.items():
            model_summary = model.summary2().tables[1]
            model_summary.to_excel(writer, sheet_name=f'{var}_model')

    # Print results
    for var, result in anova_results.items():
        print(f"ANOVA results for {var}:\n{result}\n")

    for var, model in model_results.items():
        print(f"Regression results for {var}:\n{model.summary2().tables[1]}\n")

# Example usage
csv_path = './exp_results/blobs_variable_dataset.csv'  # Input CSV file path
output_path = 'anova_model_results.xlsx'  # Output Excel file path

analyze_and_output(csv_path, output_path)
