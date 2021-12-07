from abc import ABC, abstractmethod

from mlflow import tracking

class RunEnvironment(ABC):

    def __init__(self, work_dir, project, active_run, backend_config):
        self._project = project
        self.work_dir = work_dir
        self._project = project
        self._active_run = active_run
        self.run_id = active_run.info.run_id
        self.experiment_id = active_run.info.experiment_id
        self._backend_config = backend_config

    def get_run_env_vars(self):
        """
        Returns a dictionary of environment variable key-value pairs to set in subprocess launched
        to run MLflow projects.
        """
        return {
            tracking._RUN_ID_ENV_VAR: self.run_id,
            tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
            tracking._EXPERIMENT_ID_ENV_VAR: str(self.experiment_id),
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
