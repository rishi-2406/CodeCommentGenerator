def run_unsafe_code(user_input):
    eval(user_input)
    password = "supersecretpassword123"
    print("Doing something with ", password)

def standard_function(a: int, b: int) -> int:
    return a + b
