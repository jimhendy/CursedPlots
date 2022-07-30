from functools import wraps
from typing import Any, Callable, Dict, Hashable, List

from loguru import logger


def log_all_args(log_level: str = "DEBUG") -> Callable:
    level_logger = getattr(logger, log_level.lower())

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: List[Any], **kwargs: Dict[Hashable, Any]) -> Any:
            msg = (
                f"Function called: {func.__name__}\nargs   : {args}\nkwargs : {kwargs}"
            )
            try:
                value = func(*args, **kwargs)
            except Exception as e:
                logger.error(msg)
                raise e
            msg += f"\nreturns: {value}"
            level_logger(msg)
            return value

        return wrapper

    return decorator
