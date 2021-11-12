from mlflow.entities.run import Run
from mlflow.projects import RunEnvironment

from mlflow.utils.conda import get_conda_command, get_or_create_conda_env

class CondaRunEnvironment(RunEnvironment):

    def validate_environment(self):
        self._conda_env_name = get_or_create_conda_env(self._project.conda_env_path)

    def validate_installation(self):
        pass

    def prepare_environment(self):
        pass

    def get_tracking_cmd_and_envs(self):
        pass

    def get_command(self):
        return get_conda_command(self._conda_env_name)

    @staticmethod
    def get_command_separator():
        return ' && '
