import math

TESTS_LITE = {
    '1': lambda x, y, z: x > y > z,
    '2': lambda x, y, z: x < y < z,
    '3': lambda x, y, z: x >= y >= z,
    '4': lambda x, y, z: x <= y <= z,
    '5': lambda x, y, z: x == y and y == z,
    '6': lambda x, y, z: x != y and y != z and x != z,
    '7': lambda x, y, z: x < 0 and y < 0 and z < 0,
    '8': lambda x, y, z: x + y == z,
    '9': lambda x, y, z: x * y == z,
    '10': lambda x, y, z: x < y and y > z
}

TESTS_FULL = {
    ## Easy Tests
    # Simple orders
    '1': lambda x, y, z: x > y > z,
    '2': lambda x, y, z: x < y < z,
    '3': lambda x, y, z: x >= y >= z,
    '4': lambda x, y, z: x <= y <= z,
    '5': lambda x, y, z: x < z < y,
    '6': lambda x, y, z: x <= z <= y,
    '7': lambda x, y, z: z < x < y,
    '8': lambda x, y, z: z <= x <= y,
    # Equality and Inequality
    '9': lambda x, y, z: x == y and y == z,
    '10': lambda x, y, z: x != y and y != z and x != z,
    # Signs 
    '11': lambda x, y, z: x < 0 and y < 0 and z < 0,
    '12': lambda x, y, z: x > 0 and y > 0 and z > 0,
    # Even vs Odd
    '13': lambda x, y, z: (x % 2 == 0) and (y % 2 == 0) and (z % 2 == 0),
    '14': lambda x, y, z: (x % 2 != 0) and (y % 2 != 0) and (z % 2 != 0),
    ## Medium Tests
    # Arithmetic (no division)
    '15': lambda x, y, z: x + y == z,
    '16': lambda x, y, z: x * y == z,
    '17': lambda x, y, z: x + z == y,
    '18': lambda x, y, z: x * z == y,
    '19': lambda x, y, z: y + z == x,
    '20': lambda x, y, z: y * z == x,
    # Max / Min
    '21': lambda x, y, z: max(x, y, z) == x,
    '22': lambda x, y, z: max(x, y, z) == y,
    '23': lambda x, y, z: max(x, y, z) == z,
    '24': lambda x, y, z: min(x, y, z) == x,
    '25': lambda x, y, z: min(x, y, z) == y,
    '26': lambda x, y, z: min(x, y, z) == z,
    # Sums / Products
    '27': lambda x, y, z: (x + y + z) == 0,
    '28': lambda x, y, z: (x * y * z) == 0,
    '29': lambda x, y, z: (x + y + z) % 2 == 0,
    '30': lambda x, y, z: (x + y + z) % 2 == 1,
    '31': lambda x, y, z: (x * y * z) % 2 == 0,
    '32': lambda x, y, z: (x * y * z) % 2 == 1,
    # Arithmetic Mean
    '33': lambda x, y, z: (x + y) / 2 == z,
    # Magnitude Checks
    '34': lambda x, y, z: (x <= 5 and x >= -5) and (y <= 5 and y >= -5) and (z <= 5 and z >= -5),
    '35': lambda x, y, z: (x <= 10 and x >= -10) and (y <= 10 and y >= -10) and (z <= 10 and z >= -10),
    '36': lambda x, y, z: (x <= 0 and x >= -5) and (y <= 0 and y >= -5) and (z <= 0 and z >= -5),
    '37': lambda x, y, z: (x <= 5 and x >= 0) and (y <= 5 and y >= 0) and (z <= 5 and z >= 0),
    '38': lambda x, y, z: (x <= 2 and x >= -2) and (y <= 2 and y >= -2) and (z <= 2 and z >= -2),
    '39': lambda x, y, z: (x <= 20 and x >= -20) and (y <= 20 and y >= -20) and (z <= 20 and z >= -20),
    ## Very Hard Tests
    # Pythagorean Triple
    '40': lambda x, y, z: x**2 + y**2 == z**2,
    '41': lambda x, y, z: x**2 + z**2 == y**2,
    '42': lambda x, y, z: y**2 + z**2 == x**2,
    # Bitwise Operations
    '43': lambda x, y, z: (int(x) & int(y)) == int(z),
    '44': lambda x, y, z: (int(x) | int(y)) == int(z),
    '45': lambda x, y, z: (int(x) ^ int(y)) == int(z),
    # Coprimality
    '46': lambda x, y, z: all([m % 1 == 0 for m in [x, y, z]]) and all(math.gcd(int(a), int(b)) == 1 for a, b in [(x, y), (y, z), (z, x)]),
    # Perfect Squares
    '47': lambda x, y, z: all(int(abs(n)**0.5)**2 == n for n in (x, y, z)),
    # Evil Float Cases
    '48': lambda x, y, z: (0 < x % 1) and (0 < y % 1) and (0 < z % 1),
    '49': lambda x, y, z: 0 < x % 1 < y % 1 < z % 1 < 1,
    '50': lambda x, y, z: (x < y < z) and (0 < z - x <= 1),
}

# All hypotheses based on set membership
TESTS_BAYESIAN = {
    # Rules Hypotheses
    '1': lambda x, y, z: all((n % 2 == 0) for n in (x, y, z)), # Evens
    '2': lambda x, y, z: all((n % 2 == 1) for n in (x, y, z)), # Odds
    '3': lambda x, y, z: all(int(abs(n)**0.5)**2 == n for n in (x, y, z)), # Perfect Squares
    '4': lambda x, y, z: all(math.log2(n).is_integer() for n in (x, y, z)), # Powers of 2
    '3': lambda x, y, z: all((n % 3 == 0) for n in (x, y, z)), # Multiples of 3
    '4': lambda x, y, z: all((n % 4 == 0) for n in (x, y, z)), # Multiples of 4
    '5': lambda x, y, z: all((n % 5 == 0) for n in (x, y, z)), # Multiples of 5
    '6': lambda x, y, z: all((n % 15 == 0) for n in (x, y, z)), # Multiples of 15
    # Similarity Hypotheses, Range Size 0
    '7': lambda x, y, z: all(abs(n-25) <= 0 for n in (x, y, z)),
    '8': lambda x, y, z: all(abs(n-50) <= 0 for n in (x, y, z)),
    '9': lambda x, y, z: all(abs(n-75) <= 0 for n in (x, y, z)),
    # Similarity Hypotheses, Range Size 1
    '10': lambda x, y, z: all(abs(n-25) <= 1 for n in (x, y, z)),
    '11': lambda x, y, z: all(abs(n-50) <= 1 for n in (x, y, z)),
    '12': lambda x, y, z: all(abs(n-75) <= 1 for n in (x, y, z)),
    # Similarity Hypotheses, Range Size 2
    '13': lambda x, y, z: all(abs(n-25) <= 2 for n in (x, y, z)),
    '14': lambda x, y, z: all(abs(n-50) <= 2 for n in (x, y, z)),
    '15': lambda x, y, z: all(abs(n-75) <= 2 for n in (x, y, z)),
    # Similarity Hypotheses, Range Size 4
    '16': lambda x, y, z: all(abs(n-25) <= 4 for n in (x, y, z)),
    '17': lambda x, y, z: all(abs(n-50) <= 4 for n in (x, y, z)),
    '18': lambda x, y, z: all(abs(n-75) <= 4 for n in (x, y, z)),
    # Similarity Hypotheses, Range Size 8
    '19': lambda x, y, z: all(abs(n-25) <= 8 for n in (x, y, z)),
    '20': lambda x, y, z: all(abs(n-50) <= 8 for n in (x, y, z)),
    '21': lambda x, y, z: all(abs(n-75) <= 8 for n in (x, y, z)),
    # Similarity Hypotheses, Range Size 16
    '22': lambda x, y, z: all(abs(n-25) <= 16 for n in (x, y, z)),
    '23': lambda x, y, z: all(abs(n-50) <= 16 for n in (x, y, z)),
    '24': lambda x, y, z: all(abs(n-75) <= 16 for n in (x, y, z)),
}

# All hypotheses based on set membership
TESTS_BAYESIAN_SINGLE = {
    # Rules Hypotheses
    '1': lambda x: x % 2 == 0, # Evens # 1. Odds
    '2': lambda x: x % 2 == 1, # Odds # 2. Evens
    '3': lambda x: int(abs(x)**0.5)**2 == x, # 3. Perfect Squares
    '4': lambda x: round(x ** (1/3)) ** 3 == x, # 4. Perfect Cubes
    '5': lambda x: x > 1 and all(x % i != 0 for i in range(2, int(x**0.5) + 1)), # 5. Prime numbers
    '6': lambda x: x % 3 == 0, # 6. Multiples of 3
    '7': lambda x: x % 4 == 0, # 7. Multiples of 4
    '8': lambda x: x % 5 == 0, # 8. Multiples of 5
    '9': lambda x: x % 6 == 0, # 9. Multiples of 6
    '10': lambda x: x % 7 == 0, # 10. Multiples of 7
    '11': lambda x: x % 8 == 0, # 11. Multiples of 8
    '12': lambda x: x % 9 == 0, # 12. Multiples of 9
    '13': lambda x: x % 10 == 0, # 13. Multiples of 10
    '14': lambda x: x % 11 == 0, # 14. Multiples of 11
    '15': lambda x: x % 12 == 0, # 15. Multiples of 12
    '16': lambda x: math.log2(x).is_integer(), # 16. Powers of 2
    '17': lambda x: (math.log(x) / math.log(3)).is_integer(), # 17. Powers of 3
    '18': lambda x: (math.log(x) / math.log(4)).is_integer(), # 18. Powers of 4
    '19': lambda x: (math.log(x) / math.log(5)).is_integer(), # 19. Powers of 5
    '20': lambda x: (math.log(x) / math.log(6)).is_integer(), # 20. Powers of 6
    '21': lambda x: (math.log(x) / math.log(7)).is_integer(), # 21. Powers of 7
    '22': lambda x: (math.log(x) / math.log(8)).is_integer(), # 22. Powers of 8
    '23': lambda x: (math.log(x) / math.log(9)).is_integer(), # 23. Powers of 9
    '24': lambda x: (math.log(x) / math.log(10)).is_integer(), # 24. Powers of 10
    '25': lambda x: (x - 1) % 10 == 0, # 25. Multiples of 10, +1
    '26': lambda x: (x - 2) % 10 == 0, # 26. Multiples of 10, +2
    '27': lambda x: (x - 3) % 10 == 0, # 27. Multiples of 10, +3
    '28': lambda x: (x - 4) % 10 == 0, # 28. Multiples of 10, +4
    '29': lambda x: (x - 5) % 10 == 0, # 29. Multiples of 10, +5
    '30': lambda x: (x - 6) % 10 == 0, # 30. Multiples of 10, +6
    '31': lambda x: (x - 7) % 10 == 0, # 31. Multiples of 10, +7
    '32': lambda x: (x - 8) % 10 == 0, # 32. Multiples of 10, +8
    '33': lambda x: (x - 9) % 10 == 0, # 33. Multiples of 10, +9
    # Similarity Hypotheses, Range Size 0
    '34': lambda x: abs(x-25) <= 0,
    '35': lambda x: abs(x-50) <= 0,
    '36': lambda x: abs(x-75) <= 0,
    # Similarity Hypotheses, Range Size 1
    '37': lambda x: abs(x-25) <= 1,
    '38': lambda x: abs(x-50) <= 1,
    '39': lambda x: abs(x-75) <= 1,
    # Similarity Hypotheses, Range Size 2
    '40': lambda x: abs(x-25) <= 2,
    '41': lambda x: abs(x-50) <= 2,
    '42': lambda x: abs(x-75) <= 2,
    # Similarity Hypotheses, Range Size 4
    '43': lambda x: abs(x-25) <= 4,
    '44': lambda x: abs(x-50) <= 4,
    '45': lambda x: abs(x-75) <= 4,
    # Similarity Hypotheses, Range Size 8
    '46': lambda x: abs(x-25) <= 8,
    '47': lambda x: abs(x-50) <= 8,
    '48': lambda x: abs(x-75) <= 8,
    # Similarity Hypotheses, Range Size 16
    '49': lambda x: abs(x-25) <= 16,
    '50': lambda x: abs(x-50) <= 16,
    '51': lambda x: abs(x-75) <= 16,
}
