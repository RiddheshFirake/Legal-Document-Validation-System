MOD = int(1e9 + 7)

# Function to calculate f(X) which gives the number of subsets where sum is divisible by 5
def count_subsets_divisible_by_5(X):
    dp = [0] * 5
    dp[0] = 1  # One subset (the empty subset) with sum 0

    for num in range(1, X + 1):
        new_dp = dp[:]  # Copy current state
        for i in range(5):
            new_dp[(i + num) % 5] = (new_dp[(i + num) % 5] + dp[i]) % MOD
        dp = new_dp
    
    return dp[0]

# Main function to handle the array and sum of results
def complex_counting_hard_version(arr):
    total_sum = 0
    for X in arr:
        total_sum = (total_sum + count_subsets_divisible_by_5(X)) % MOD
    return total_sum

# Input
N = int(input())  # Length of the array
A = list(map(int, input().split()))  # Array elements

# Output the result
result = complex_counting_hard_version(A)
print(f"DJSISACA{{{result}}}")
