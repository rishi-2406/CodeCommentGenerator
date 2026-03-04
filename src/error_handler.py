"""
Error handling module for the Code Comment Generation pipeline.
Defines custom exceptions and error formatting utilities.
"""


class ParserError(Exception):
    """Custom exception for parsing and validation errors."""
    def __init__(self, message: str, line: int = None, column: int = None):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(message)


class ExtractionError(Exception):
    """Raised when AST feature extraction fails."""
    def __init__(self, message: str, node_type: str = None):
        self.message = message
        self.node_type = node_type
        super().__init__(message)


class CommentGenerationError(Exception):
    """Raised when comment generation fails for a node."""
    def __init__(self, message: str, node_id: str = None):
        self.message = message
        self.node_id = node_id
        super().__init__(message)


def format_error(error: Exception) -> str:
    """Formats any pipeline error into a readable string."""
    if isinstance(error, ParserError):
        location = ""
        if error.line is not None:
            location = f" at line {error.line}"
            if error.column is not None:
                location += f", column {error.column}"
        return f"Error{location}: {error.message}"
    elif isinstance(error, ExtractionError):
        node_info = f" (node: {error.node_type})" if error.node_type else ""
        return f"ExtractionError{node_info}: {error.message}"
    elif isinstance(error, CommentGenerationError):
        node_info = f" (node_id: {error.node_id})" if error.node_id else ""
        return f"CommentGenerationError{node_info}: {error.message}"
    else:
        return f"Error: {str(error)}"
