import sys
from typing import Optional


def _format_error_details(error: BaseException, error_detail: sys) -> str:
    """
    Extracts useful debug information: file name, line number, and error message.
    """
    exc_type, exc_obj, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # Fallback when traceback is not available
        return f"Error: {str(error)}"

    tb = exc_tb
    while tb.tb_next is not None:
        tb = tb.tb_next
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno

    return (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )


class CustomException(Exception):
    """
    Custom exception wrapper used across the project to provide consistent,
    traceable error messages.

    Usage:
        try:
            ...
        except Exception as e:
            raise CustomException(e, sys)
    """

    def __init__(self, error: BaseException, error_detail: sys):
        self.original_exception: Optional[BaseException] = error
        self.error_message: str = _format_error_details(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message
