import logging
import os
import posixpath
import shutil
import tempfile
from typing import final
import urllib.parse
import urllib.request

from mlflow.exceptions import MlflowException

from spython.utils import check_install
from spython.main import Client

from mlflow import tracking
from mlflow.exceptions import ExecutionException
from mlflow.projects import RunEnvironment
from mlflow.projects.utils import MLFLOW_CONTAINER_WORKDIR_PATH, is_user_admin
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.utils import process, file_utils
from mlflow.utils.mlflow_tags import MLFLOW_SINGULARITY_IMAGE_URI, MLFLOW_SINGULARITY_IMAGE_ID

from mlflow.projects.utils import (
    get_databricks_env_vars,
    get_local_uri_or_none,
    convert_container_args_to_list,
    make_volume_abs,
    make_path_abs,
    get_paths_to_ignore,
    root_temp_directory,
    PROJECT_SINGULARITY_ARGS,
    MLFLOW_CONTAINER_WORKDIR_PATH
)


_logger = logging.getLogger(__name__)

_GENERATED_RECIPE_NAME = "Singularity.mlflow-autogenerated"
_MLFLOW_SINGULARITY_TRACKING_DIR_PATH = "/mlflow/tmp/mlruns"
_PROJECT_TAR_ARCHIVE_NAME = "mlflow-project-singularity-build-context"


class SingularityRunEnvironment(RunEnvironment):

    def validate_installation(self):
        """
        Verify if Singularity is installed on host machine.
        """
        if not check_install():
            raise ExecutionException(
                "Could not find Singularity executable. "
                "Ensure Singularity is installed as per the instructions "
                "at https://sylabs.io/guides/3.3/user-guide/installation.html."
            )

    def validate_environment(self):
        if not self._project.name:
            raise ExecutionException(
                "Project name in MLProject must be specified when using singularity " "for image tagging."
            )
        if not self._project.singularity_env.get("image"):
            raise ExecutionException(
                "Project with singularity environment must specify the singularity image "
                "to use via an 'image' field under the 'singularity_env' field."
            )

    def prepare_environment(self):
        """
        Build a Singularity image containing the project in `work_dir`, using the base image.
        """

        image_uri = self._project.name + '.sif'

        # Bootstrap type varies based on base image
        bootstrap = "localimage"
        base_image = self._project.singularity_env.get("image")
        recipe_image = base_image
        if base_image.startswith('library://'):
            bootstrap = "library"
        elif base_image.startswith('shub://'):
            bootstrap = "shub"
        elif base_image.startswith('docker://'):
            bootstrap = "docker"
        else:
            recipe_image = os.path.join(self.work_dir, base_image)
            if not os.path.exists(recipe_image):
                raise ExecutionException(
                    "Base image in project working directory not found: %s" % base_image)
        # Default to current working directory if no build dir is specified
        build_dir = self._project.singularity_env.get("build_dir", ".")
        if not os.path.isabs(build_dir):
            build_dir = make_path_abs(build_dir)
        final_image = os.path.join(self.work_dir, build_dir, image_uri)

        _logger.info('Preparing Singularity image.')
        if os.path.exists(final_image):
            _logger.info(
                "Final image %s already exists in working directory, reusing." % final_image)
        else:
            _logger.info(f'Final image does not exist locally. Pulling {base_image}...')
            Client.pull(base_image, pull_folder=build_dir, name=final_image)

        self._image = final_image

    def _get_singularity_image_uri(self):
        """
        Returns an appropriate Docker image URI for a project based on the git hash of the specified
        working directory.

        :param repository_uri: The URI of the Docker repository with which to tag the image. The
                            repository URI is used as the prefix of the image URI.
        :param work_dir: Path to the working directory in which to search for a git commit hash
        """
        
        # NOTE: This is not used anymore, we always only use the base image
        # without adding new layers
        """
        repository_uri = self._project.name
        repository_uri = repository_uri if repository_uri else "singularity-project"
        # Optionally include first 7 digits of git SHA in tag name, if available.
        git_commit = _get_git_commit(self.work_dir)
        version_string = ":" + git_commit[:7] if git_commit else ""
        return repository_uri + version_string + ".sif"
        """

    def _create_singularity_build_ctx(self, tmp_directory, recipe_contents):
        """
        Creates build context containing Singularity recipe and project code, returning path to folder
        """
        ignore_func = shutil.ignore_patterns(
            *get_paths_to_ignore(self.work_dir))
        try:
            dst_path = os.path.join(tmp_directory, "mlflow-project-contents")
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src=self.work_dir, dst=dst_path,
                            ignore=ignore_func)
            with open(os.path.join(dst_path, _GENERATED_RECIPE_NAME), "w") as handle:
                handle.write(recipe_contents)
        except Exception as exc:
            print(exc)
            raise ExecutionException(
                "Issue creating Singularity build context at %s" % dst_path
            )
        return dst_path

    def get_command(self):
        cmd = []
        cmd = Client._init_command("exec", cmd)
        singularity_args = self._backend_config[PROJECT_SINGULARITY_ARGS]
        cmd = convert_container_args_to_list(cmd, singularity_args)
        volumes = self._project.singularity_env.get("volumes")
        if volumes:
            # Hack so that we can use relative paths and variables in the volumes to mount
            for i, volume in enumerate(volumes):
                volumes[i] = make_volume_abs(volume)
            cmd += Client._generate_bind_list(volumes)

        # TODO: include singularity options too?

        env_vars = self.get_run_env_vars()
        tracking_uri = tracking.get_tracking_uri()
        tracking_cmds, tracking_envs = self._get_tracking_cmd_and_envs()
        artifact_cmds, artifact_envs = self._get_artifact_storage_cmd_and_envs()

        cmd += tracking_cmds + artifact_cmds
        env_vars.update(tracking_envs)
        env_vars.update(artifact_envs)

        # This could also be consolidated between two container technologies
        user_env_vars = self._project.singularity_env.get("environment")
        if user_env_vars is not None:
            for user_entry in user_env_vars:
                if isinstance(user_entry, list):
                    # User has defined a new environment variable for the docker environment
                    env_vars[user_entry[0]] = user_entry[1]
                else:
                    # User wants to copy an environment variable from system environment
                    system_var = os.environ.get(user_entry)
                    if system_var is None:
                        raise MlflowException(
                            "This project expects the %s environment variables to "
                            "be set on the machine running the project, but %s was "
                            "not set. Please ensure all expected environment variables "
                            "are set" % (", ".join(user_env_vars), user_entry)
                        )
                    env_vars[user_entry] = system_var

        for key, value in env_vars.items():
            cmd += ["--env", "{key}={value}".format(key=key, value=value)]
        # TODO we have to set the working directory as an argument on the command tring
        # because the new image does not necessarily get built (which would also set
        # the working directory accordingly) when the user doesn't have admin rights
        cmd += ["--pwd", MLFLOW_CONTAINER_WORKDIR_PATH]
        cmd += [self._image]
        return cmd

    def _get_tracking_cmd_and_envs(self):
        cmds = []
        env_vars = dict()

        tracking_uri = tracking.get_tracking_uri()
        local_path, container_tracking_uri = get_local_uri_or_none(
            tracking_uri)
        if local_path is not None:
            cmds = ["--bind", "%s:%s" %
                    (local_path, _MLFLOW_SINGULARITY_TRACKING_DIR_PATH)]
            env_vars[tracking._TRACKING_URI_ENV_VAR] = container_tracking_uri
        env_vars.update(get_databricks_env_vars(tracking_uri))
        return cmds, env_vars

    def _get_artifact_storage_cmd_and_envs(self):
        return [], {}

    def add_run_args(self, run_args):
        singularity_args = self._backend_config[PROJECT_SINGULARITY_ARGS]
        for key, value in singularity_args.items():
            args = key if isinstance(value, bool) else "%s=%s" % (key, value)
            run_args.extend(["--singularity-args", args])
        return run_args

    @staticmethod
    def get_command_separator():
        return ' '
