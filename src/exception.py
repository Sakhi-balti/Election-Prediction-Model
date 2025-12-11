import sys

def get_error_detail(error_message, error_detail):
    exc_type, exc_value, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {error_message}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script: {file_name}\n"
        f"At line number: {line_number}\n"
        f"Message: {error_message}"
    )

    
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error = get_error_detail(error_message, error_detail)
    def __str__(self):
        return self.error    
