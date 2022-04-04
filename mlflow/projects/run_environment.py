from abc import ABC, abstractmethod

from mlflow import tracking

class RunEnvironment(ABC):

    def __init__(self, work_dir, project, backend_config):
        self._project = project
        self.work_dir = work_dir
        self._project = project
        self._backend_config = backend_config

    def get_run_env_vars(self):
        """
        Returns a dictionary of environment variable key-value pairs to set in subprocess launched
        to run MLflow projects.
        """
        return {
            tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri()
        }

    @abstractmethod
    def validate_environment(self):
        pass

    @abstractmethod
    def validate_installation(self):
        pass

    @abstractmethod
    def prepare_environment(self):
        pass

    @abstractmethod
    def get_command(self):
        pass
    
    @abstractmethod
    def add_run_args(self, run_args):
        pass

    @staticmethod
    def get_command_separator():
        raise NotImplementedError
