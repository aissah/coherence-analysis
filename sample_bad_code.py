def factorial(n): f = 1 for i in range(1, n+1): f = f * i return f

def fib(n):
    if n<=1: return n
    else: return(fib(n-1) + fib(n-2))

def gcd(a,b): while b != 0: temp = b b = a % b a = temp return a

def primes(n): p_list = [] for i in range(2, n+1): prime = True for j in range(2, int(i ** 0.5) + 1): if i % j == 0: prime = False break if prime: p_list.append(i) return p_list

a = 5
b = 10
print("Factorial of ", a, " is: ", factorial(a))
print("Factorial of ", b, " is: ", factorial(b))
print("First ", a, " numbers of fibonacci sequence: ")
for i in range(a): print(fib(i))

print("GCD of ", a, " and ", b, " is: ", gcd(a, b))

prime_numbers = primes(100)
print("Prime numbers up to 100: ")
for p in prime_numbers: print(p)

def is_palindrome(string): string = string.lower() rev_str = string[::-1] if string == rev_str: return True else: return False

s = "Racecar"
if is_palindrome(s): print(s, " is a palindrome!")
else: print(s, " is not a palindrome!")

def bubbleSort(arr): n = len(arr) for i in range(n-1): for j in range(0, n-i-1): if arr[j] > arr[j+1] : arr[j], arr[j+1] = arr[j+1], arr[j]

arr = [64, 25, 12, 22, 11]
bubbleSort(arr)
print ("Sorted array is: ")
for i in range(len(arr)): print ("%d" %arr[i]),

def linear_search(arr, x):
    for i in range(len(arr)): if arr[i] == x: return i
    return -1

arr = [2, 3, 4, 10, 40]
x = 10
result = linear_search(arr, x)
if result != -1: print("Element is present at index", result)
else: print("Element is not present in array")

def reverse_string(s): return s[::-1]
str1 = "Hello World"
print("Reversed string: ", reverse_string(str1))

