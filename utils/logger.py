"""
Module for managing logging in a project.

This module defines a `_Logger` class that provides methods to create and manage
loggers for different parts of the application. It allows for flexible logging
capabilities, including logging to files and to the console, with log directories
and loggers that can be configured dynamically.

The module includes the following features:
- Create loggers with specified names.
- Configure the log directory for log files.
- Set and manage the active logger for logging messages.
- Supports logging to both files and the console with custom formats.
- Provides methods to access and use the active logger dynamically.

Classes:
    _Logger: A class to manage and handle multiple loggers with file and stream outputs.

Example usage:
    # Create a logger
    logger = _Logger()
    logger.set_logdir("path/to/log/directory")
    my_logger = logger.create_logger("my_logger")

    # Set active logger and log messages
    logger.set_logger("my_logger")
    logger.info("This is an info message.")
    logger.error("This is an error message.")
"""

import logging
import os
from typing import Any


class _Logger:
    """A class to manage and handle multiple loggers with file and stream outputs.

    This class provides functionality to create loggers, configure log directories,
    and set an active logger for logging messages. It also supports managing loggers
    with different names and log levels.

    Attributes:
        _loggers (dict): A dictionary that holds created loggers by their names.
        _active_logger (str or None): The name of the currently active logger.
        _log_dir (str or None): The directory where log files are stored.
    """

    def __init__(self):
        """Initializes the _Logger instance with empty loggers, no active logger,
        and no log directory.

        This constructor is called automatically when an instance of _Logger is created.
        """
        self._loggers = {}
        self._active_logger = None
        self._log_dir = None

    def create_logger(self, name: str) -> logging.Logger:
        """Creates a new logger with the specified name.

        This method initializes a new logger instance with the specified name and
        adds both a file handler and a stream handler. The logger writes to both
        a log file in the specified log directory and the console.

        Args:
            name (str): The name of the logger to be created.

        Returns:
            logging.Logger: The created logger instance.

        Raises:
            ValueError: If a logger with the same name already exists.
            Exception: If `set_logdir` has not been called before creating the logger.
        """
        if name in self._loggers:
            raise ValueError(f"logger:'{name}' exists.")

        if self._log_dir is None:
            raise Exception("call `set_logdir` before creating logger.")

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
        file_handler = logging.FileHandler(os.path.join(self._log_dir, f"{name}.log"))
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(fmt)
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self._loggers[name] = logger
        self.__setattr__(name, logger)

        return logger

    def set_logdir(self, log_dir: str) -> None:
        """Sets the directory where all log files will be stored.

        This method should be called before creating any logger. The directory
        will be used for all loggers created afterward. If the directory does not
        exist, it will be created.

        Args:
            log_dir (str): The directory where log files will be saved.

        Raises:
            AssertionError: If log files already exist or log directory is set.
        """
        assert self._log_dir is None and len(self._loggers) == 0

        if not os.path.exists(log_dir):
            try:
                os.makedirs(os.path.abspath(log_dir))
            except:
                pass

        self._log_dir = log_dir

    def set_logger(self, name: str) -> logging.Logger:
        """Sets the active logger by its name.

        This method allows the user to set a specific logger as the active logger
        so that log messages can be recorded using that logger.

        Args:
            name (str): The name of the logger to be set as the active logger.

        Returns:
            logging.Logger: The logger instance that is now the active logger.

        Raises:
            ValueError: If the specified logger does not exist.
        """
        if name not in self._loggers:
            self.create_logger(name)
        self._active_logger = name
        return self._loggers[self._active_logger]

    def __getattribute__(self, __name: str) -> Any:
        """Dynamically accesses the methods or attributes of the active logger.

        This method allows access to methods of the active logger as if they were
        attributes of the _Logger class. If no active logger is set, an exception is raised.

        Args:
            __name (str): The name of the attribute or method being accessed.

        Returns:
            Any: The value of the attribute or the return value of the method.

        Raises:
            NotImplementedError: If no active logger has been set yet.
            AttributeError: If the specified attribute or method is not found.
        """
        try:
            return object.__getattribute__(self, __name)
        except:
            try:
                if self._active_logger is None:
                    raise NotImplementedError(
                        f"No logger available. Call `create_logger` or `set_logger` to initialize a logger."
                    )
                return getattr(self._loggers[self._active_logger], __name)
            except:
                raise

    def logdir(self) -> str:
        """Returns the current log directory.

        This method provides the path to the directory where the log files are stored.

        Returns:
            str: The path to the log directory.

        Raises:
            AttributeError: If the log directory has not been set yet.
        """
        return self._log_dir


logger = _Logger()
