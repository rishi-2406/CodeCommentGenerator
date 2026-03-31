# SECURITY WARNING: Uses dangerous builtin 'eval'
# SECURITY WARNING: Potential hardcoded secret assigned to 'password'
def run_unsafe_code(user_input):
    """
    Run unsafe code on the given user input.
    
    Security Warnings:
        - Uses dangerous builtin 'eval'
        - Potential hardcoded secret assigned to 'password'
    """
    eval(user_input)
    password = "supersecretpassword123"
    print("Doing something with ", password)


def standard_function(a: int, b: int) -> int:
    """
    Returns a standard function.
    """
    return a + b