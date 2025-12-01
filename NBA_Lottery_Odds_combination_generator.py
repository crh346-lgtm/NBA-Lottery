# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:19:14 2025

@author: 14153
"""

from itertools import combinations
import pandas as pd
import random
import Theoretical_weighted_ping_pong_prob as NBA
import numpy as np
import itertools
import NBA_Lottery_Shannon_Free as Hill
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

numbers = range(1, 15)  # 1 to 14 inclusive

def create_lottery_combos():
# Define the range of numbers
    # numbers = range(1, 15)  # 1 to 14 inclusive
    
    # Generate all 4-number combinations
    all_combinations = list(combinations(numbers, 4))
    return all_combinations

def create_combo_df(combo_list, odds_list):
    data_list=odds_list
    num=pd.DataFrame(combo_list)
    y=0
    for x in range(0, len(data_list)):
        # print(x)
        num.loc[y:y+data_list[x], 'draft_position'] = int(numbers[x])
        y+=data_list[x]
    num = num.rename(columns={0: "Num1", 1: "Num2", 2: "Num3",3:"Num4"})
    return num

def sum_individual_numbers(num):
    result = {}
    other_cols = [c for c in num.columns if c != 'draft_position']
    for i, val in enumerate(num['draft_position'].unique(), start=1):
        # Subset rows where main_col == val
        subset = num[num['draft_position'] == val][other_cols]
        
        # Flatten all values into a single list
        values = subset.values.flatten()
        
        # Count how many times each number 1–14 appears
        counts = {num: (values == num).sum() for num in range(1, 15)}
        
        result[i] = counts
    res_df=pd.DataFrame(result)
    cols = res_df.columns.tolist()
    cols_new=['Team '+str(x) for x in cols]
    res_df.columns=cols_new
    # res_df.to_csv(r'G:\My Drive\NBA_Sloan\lottery_combo_sums.csv', index=True)

    # res_t=res_df.T
    # print(res_df)
    return res_df

def calculate_percent(df,r_c=False):
    if r_c==True:
        row_sums = df.sum(axis=0)
        df_normalized = df.div(row_sums, axis=1)
    row_sums = df.sum(axis=1)
    df_normalized = df.div(row_sums, axis=0)
    return df_normalized

def randomize_combinations(combos, odds_list):
    combos=combos[:1000]
    random_df=pd.DataFrame([],columns=['Num1', 'Num2','Num3','Num4', 
                                       'draft_position'])
    z=1
    for x in odds_list:
        o_list=random.sample(combos, x)
        for y in o_list:
            combos.remove(y)
        new_df=pd.DataFrame(o_list, columns=['Num1', 'Num2','Num3','Num4'])
        new_df['draft_position']=z
        z+=1
        random_df=pd.concat([random_df, new_df])
        random_df=random_df.sort_values(by='Num1')
    return random_df
# def calculate_percent_col(df)
# for x in numbers:
#     test=num[num['draft_position']==x]
    
def calculate_permutation_probability(row, weights):
    """
    Calculates the mathematically accurate probability of a 4-number combination 
    being drawn under weighted, no-replacement sampling.
    This function is applied row-wise (axis=1).

    Args:
        row (pd.Series): A single row of the DataFrame, containing Num1 to Num4.
        weights (dict): The current weight mapping.

    Returns:
        float: The total probability of that combination being drawn.
    """
    
    # 1. Extract the four numbers for this row
    combination = (row['Num1'], row['Num2'], row['Num3'], row['Num4'])
    
    P_combination = 0.0
    sum_of_weights = sum(weights.values())
    
    # 2. Iterate over all 24 possible orderings (permutations) of the combination
    for ordered_draw in itertools.permutations(combination):
        P_ordered_draw = 1.0
        
        # Simulate the drawing process sequentially (conditional probability)
        temp_weights = weights.copy()
        temp_sum_weights = sum_of_weights
        
        for number_drawn in ordered_draw:
            # Probability of drawing this number_drawn at this step
            P_draw = temp_weights[number_drawn] / temp_sum_weights
            P_ordered_draw *= P_draw
            
            # Update weights for the next draw (NO REPLACEMENT)
            temp_sum_weights -= temp_weights[number_drawn]
            del temp_weights[number_drawn] # Remove the drawn number from the pool
        
        # Add the probability of this specific ordered sequence to the combination's total
        P_combination += P_ordered_draw
        
    return P_combination

def calculate_accurate_combos_normalized(combo_df_filtered, weights):
    """
    Calculates the accurate probability for each combination in the filtered 
    DataFrame and then normalizes them so the total sum is 1.0.
    """
    
    required_cols = ['Num1', 'Num2', 'Num3', 'Num4']
    if not all(col in combo_df_filtered.columns for col in required_cols):
        raise ValueError("DataFrame must contain 'Num1', 'Num2', 'Num3', and 'Num4' columns.")

    print("1. Calculating raw accurate probabilities...")
    # Apply the row-wise calculation function to get RAW probability
    combo_df_filtered['Raw_Accurate_Prob'] = combo_df_filtered.apply(
        lambda row: calculate_permutation_probability(row, weights), 
        axis=1
    )
    
    # 2. Re-normalization Step 🏆
    # The sum of all raw probabilities must be less than 1.0 (since one was excluded).
    total_raw_prob_remaining = combo_df_filtered['Raw_Accurate_Prob'].sum()
    
    print(f"2. Sum of raw probabilities for the remaining {len(combo_df_filtered)} combos: {total_raw_prob_remaining:.6f}")
    
    # New Normalized Probability = Raw Probability / Total Raw Probability Remaining
    combo_df_filtered['Normalized_Accurate_Prob'] = (
        combo_df_filtered['Raw_Accurate_Prob'] / total_raw_prob_remaining
    )

    # 3. Final Sanity Check
    final_sum_check = combo_df_filtered['Normalized_Accurate_Prob'].sum()
    print(f"3. Final sum of Normalized Probabilities: {final_sum_check:.6f} (Should be 1.000000)")

    return combo_df_filtered

def calculate_accurate_combos(combo_df, weights):
    """
    Calculates the accurate probability for each 4-number combination 
    by applying the permutation calculation row-by-row.
    """
    combo_df2=combo_df.copy()
    # Check if the required columns exist
    required_cols = ['Num1', 'Num2', 'Num3', 'Num4']
    if not all(col in combo_df2.columns for col in required_cols):
        raise ValueError("DataFrame must contain 'Num1', 'Num2', 'Num3', and 'Num4' columns.")

    # 1. Apply the row-wise calculation function
    # axis=1 tells pandas to pass each row (as a Series) to the function
    print("Calculating accurate probabilities (this may take time for many rows)...")
    combo_df2['Accurate_Prob'] = combo_df2.apply(
        lambda row: calculate_permutation_probability(row, weights), 
        axis=1
    )
    
    # 2. Sanity Check
    # Since this DataFrame only represents a subset of the 1001 combinations,
    # we check the sum of probabilities *within this subset*.
    total_subset_prob = combo_df2['Accurate_Prob'].sum()
    print(f"\nSum of probabilities for this subset of {len(combo_df2)} rows: {total_subset_prob:.6f}")

    return combo_df2


# def get_weighted_probs(df,weights):
    

def create_probability_summary(df):
    """
    Groups the DataFrame by 'draft_position' and sums the 
    'Normalized_Accurate_Prob' column to create a new summary DataFrame.
    """
    
    # 1. Group by 'draft_position' and calculate the sum of the probability column
    df_prob_sums = df.groupby('draft_position')[
        'Normalized_Accurate_Prob'
    ].sum().reset_index()

    # 2. Rename the aggregated column for clarity
    df_prob_sums.rename(
        columns={'Normalized_Accurate_Prob': 'Total_Position_Probability'}, 
        inplace=True
    )
    
    # 3. Sort the results by draft_position for a clean presentation
    df_prob_sums.sort_values(by='draft_position', inplace=True)
    
    return df_prob_sums

def bar_chart_with_weight_diffs(norm_df,list_of_weight_dicts):
    df_list=[]
    for x in list_of_weight_dicts:
        df_list.append(create_probability_summary(calculate_accurate_combos_normalized(norm_df, x)))
    return df_list
        
    # Norm_weights=calculate_accurate_combos_normalized(test, weights)
    # Hill_norm=calculate_accurate_combos_normalized(Hill_df, weights)
    # unique_draft_positions = Norm_weights['draft_position'].unique()
    # uni2= Hill_df['draft_position'].unique()
    # for x in unique_draft_positions:
    #     temp=Norm_weights[Norm_weights['draft_position']==x]
    #     print(temp['Normalized_Accurate_Prob'].sum())
    # for y in uni2:
    #     temp2=Hill_norm[Hill_norm['draft_position']==y]
    #     print(temp2['Normalized_Accurate_Prob'].sum())
    # # Norm_weights_t1=Norm_weights[Norm_weights['draft_position']==1.0]
    # # print(Norm_weights_t1['Normalized_Accurate_Prob'].sum())
    # low_high=calculate_accurate_combos_normalized(test, weights_1_high)
    # Hill_diff=calculate_accurate_combos_normalized(Hill_df, weights_1_high)
    # for x in unique_draft_positions:
    #     temp=low_high[Norm_weights['draft_position']==x]
    #     print(temp['Normalized_Accurate_Prob'].sum())
    # for y in uni2:
    #     temp2=Hill_diff[Hill_diff['draft_position']==y]
    #     print(temp2['Normalized_Accurate_Prob'].sum())
    return


if __name__=='__main__':
    data_list=[140,140,140,125,105,90,75,60,45,30,20,15,10,5]
    pre_2019_odds=[250,199,156,119,88,63,43,28,17,11,8,7,6,5]
    weights= {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_1_2={1:2,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_1_10={1:10,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_1_h={1:.5,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_1_tnth={1:.1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    
    weights= {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_2_3_2={1:1,2:2,3:2,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_2_3_10={1:1,2:10,3:10,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_2_3_h={1:1,2:.5,3:.5,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    weights_2_3_tnth={1:1,2:.1,3:.1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1}
    
        
    a=create_combo_df(create_lottery_combos(), data_list)
    Hill_df=Hill.randomize_secrets(create_lottery_combos(), data_list)
    # Hill_df.to_csv(r'G:\My Drive\NBA_sloan\Data\CSPRNG_Gendata_Ball_2_3_V2.csv',index=False)
    Hill_df=Hill_df.sort_values(by='draft_position')
    test=a.iloc[:1000]
    b=sum_individual_numbers(a)
    c=calculate_percent(b)
    d=b.T
    e=c.T
    f=create_combo_df(create_lottery_combos(), pre_2019_odds)
    g=sum_individual_numbers(f)
    h=calculate_percent(g)
    i=h.T
    j=randomize_combinations(create_lottery_combos(), data_list)
    # j.to_csv(r'G:\My Drive\NBA_Sloan\lottery_randomization_table.csv', index=True)
    k=sum_individual_numbers(j)
    l=calculate_percent(k)
    m=k.T
    n=l.T
    o=n-e
    # p=a.iloc[]
    # q=p.drop('draft_position', axis=1)
    # test=a[a['draft_position']==1.0]
    # r=NBA.prob_wrapper(a[a['draft_position']==1.0],list(range(1,15)),np.array([1] +[1]*13))
    t=calculate_accurate_combos(test, weights)
    # Norm_weights=calculate_accurate_combos_normalized(test, weights)
    # Hill_norm=calculate_accurate_combos_normalized(Hill_df, weights)
    # unique_draft_positions = Norm_weights['draft_position'].unique()
    # uni2= Hill_df['draft_position'].unique()
    # for x in unique_draft_positions:
    #     temp=Norm_weights[Norm_weights['draft_position']==x]
    #     print(temp['Normalized_Accurate_Prob'].sum())
    # for y in uni2:
    #     temp2=Hill_norm[Hill_norm['draft_position']==y]
    #     print(temp2['Normalized_Accurate_Prob'].sum())
    # # Norm_weights_t1=Norm_weights[Norm_weights['draft_position']==1.0]
    # # print(Norm_weights_t1['Normalized_Accurate_Prob'].sum())
    # # low_high=calculate_accurate_combos_normalized(test, weights_1_high)
    # # Hill_diff=calculate_accurate_combos_normalized(Hill_df, weights_1_high)
    # for x in unique_draft_positions:
    #     temp=low_high[Norm_weights['draft_position']==x]
    #     print(temp['Normalized_Accurate_Prob'].sum())
    # for y in uni2:
    #     temp2=Hill_diff[Hill_diff['draft_position']==y]
    #     print(temp2['Normalized_Accurate_Prob'].sum())
    # try1=create_probability_summary(low_high)    
    
    colors=['#2166AC', '#67A9CF', '#000000','#EF8A62','#B2182B']
    legend=['Ball 1:10%', 'Ball 1:90%','Ball 1:100%','Ball 1:200%','Ball 1:1000%']
    legend2=['Ball 2,3:10%', 'Ball 2,3:90%','Ball 2,3:100%','Ball 2,3:200%','Ball 2,3:1000%']
    w2=[weights_2_3_tnth,weights_2_3_h,weights,weights_2_3_2,weights_2_3_10]
    w1=[weights_1_tnth,weights_1_h,weights,weights_1_2,weights_1_10]
    labels=list(range(1,15))
    positions = [x + 0.3 for x in labels]
    # positions=list(int((range(1.3,15.3))))
    # plt.figure()
    # vac=bar_chart_with_weight_diffs(test, w2)
    # n=0
    # # plt.figure()
    # fig, ax = plt.subplots()
    # for x in range(0, len(vac)):
    #     plt.bar(x=vac[x]['draft_position']+n,height=vac[x]['Total_Position_Probability'],width=0.15,color=colors[x],label=legend[x])
    #     n+=.15
    # plt.title('Change in #1 Pick Probablity with Sequential Combination Assignment')
    # plt.ylabel('Probability of #1 Pick Selection')
    # ax.set_ylim(0, 0.7)
    # plt.xlabel('Lottery Odds Order (Team)')
    # plt.xticks(positions,labels)
    # # ax.xaxis.set_major_locator(MultipleLocator(1))
    # plt.legend()
    # file_name=r'G:\My Drive\NBA_Sloan\Graphs\Graphs_standard_lottery_sequential_V2_2_3.png'
    # plt.savefig(file_name, dpi=500, bbox_inches='tight')
    # # plt.close()
    
    # plt.figure()
    fig, ax2 = plt.subplots()
    vac=bar_chart_with_weight_diffs(Hill_df, w2)
    n=0
    for x in range(0, len(vac)):
        plt.bar(x=vac[x]['draft_position']+n,height=vac[x]['Total_Position_Probability'],width=0.15,color=colors[x],label=legend2[x])
        n+=.15
    plt.title('Change in #1 Pick Probablity with CSPRNG Combination Assignment')
    plt.ylabel('Probability of #1 Pick Selection')
    ax2.set_ylim(0, .7)
    plt.xlabel('Lottery Odds Order (Team)')
    plt.xticks(positions,labels)
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.legend()
    file_name2=r'G:\My Drive\NBA_Sloan\Graphs\Graphs_CSPRNG_lottery_V2_2_3.png'
    # file_name=r'G:\My Drive\NBA_Sloan\Graphs\Graphs_lottery_CSPRNG_V2_1.png'
    plt.savefig(file_name2, dpi=500, bbox_inches='tight')
        # plt.bar(x=vac[1]['draft_position']+.3,height=vac[1]['Total_Position_Probability'],width=0.1)
    # for x in vac:
    #     plt.bar(x=x['draft_position']+n,height=x['Total_Position_Probability'],width=0.3)
    #     n+=2
    # chart_df=pd.DataFrame()
    
    # Norm_weights_t1=Norm_weights[Norm_weights['draft_position']==1.0]
    # print(Norm_weights_t1['Normalized_Accurate_Prob'].sum())
    
    # k.to_csv(r'G:\My Drive\NBA_Sloan\lottery_ramdon_combo_sum.csv', index=True)
    # l.to_csv(r'G:\My Drive\NBA_Sloan\lottery_ramdon_combo_sum_percent.csv', index=True)
    # m.to_csv(r'G:\My Drive\NBA_Sloan\lottery_ramdon_combo_sum_transform.csv', index=True)
    # n.to_csv(r'G:\My Drive\NBA_Sloan\lottery_ramdon_combo_sum_percent_transform.csv', index=True)
    # o.to_csv(r'G:\My Drive\NBA_Sloan\lottery_odds_delta_current_vs_1_randomization.csv', index=True)
