from codecs import ignore_errors
import logging
import os
import posixpath
import shutil
import tempfile
import urllib.parse
import urllib.request

import docker

from mlflow.exceptions import MlflowException

from mlflow import tracking
from mlflow.projects import RunEnvironment, artifact_storage
from mlflow.exceptions import ExecutionException
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.utils import process, file_utils
from mlflow.utils.mlflow_tags import MLFLOW_DOCKER_IMAGE_URI, MLFLOW_DOCKER_IMAGE_ID
from mlflow.projects.utils import (
    convert_container_args_to_list,
    ignore_patterns,
    make_volume_abs,
    get_databricks_env_vars,
    get_local_uri_or_none,
    get_paths_to_ignore,
    root_temp_directory,
    ignore_patterns,
    PROJECT_DOCKER_ARGS,
    MLFLOW_CONTAINER_TRACKING_DIR_PATH,
    MLFLOW_CONTAINER_WORKDIR_PATH
)

_logger = logging.getLogger(__name__)

_GENERATED_DOCKERFILE_NAME = "Dockerfile.mlflow-autogenerated"
_PROJECT_TAR_ARCHIVE_NAME = "mlflow-project-docker-build-context"


class DockerRunEnvironment(RunEnvironment):

    def validate_installation(self):
        """
        Verify if Docker is installed on host machine.
        """
        try:
            docker_path = "docker"
            process.exec_cmd([docker_path, "--help"], throw_on_error=False)
        except EnvironmentError:
            raise ExecutionException(
                "Could not find Docker executable. "
                "Ensure Docker is installed as per the instructions "
                "at https://docs.docker.com/install/overview/."
            )

    def validate_environment(self):
        if not self._project.name:
            raise ExecutionException(
                "Project name in MLProject must be specified when using docker " "for image tagging."
            )
        if not self._project.docker_env.get("image"):
            raise ExecutionException(
                "Project with docker environment must specify the docker image "
                "to use via an 'image' field under the 'docker_env' field."
            )

    def prepare_environment(self):
        """
        Build a docker image containing the project in `work_dir`, using the base image.
        """
        image_uri = self._get_docker_image_uri()
        base_image = self._project.docker_env.get("image")
        dockerfile = (
            "FROM {imagename}\n" "COPY {build_context_path}/ {workdir}\n" "WORKDIR {workdir}\n"
        ).format(
            imagename=base_image,
            build_context_path=_PROJECT_TAR_ARCHIVE_NAME,
            workdir=MLFLOW_CONTAINER_WORKDIR_PATH,
        )
        build_ctx_path = self._create_docker_build_ctx(dockerfile)
        with open(build_ctx_path, "rb") as docker_build_ctx:
            _logger.info("=== Building docker image %s ===", image_uri)
            client = docker.from_env()
            image, _ = client.images.build(
                tag=image_uri,
                forcerm=True,
                dockerfile=posixpath.join(_PROJECT_TAR_ARCHIVE_NAME, _GENERATED_DOCKERFILE_NAME),
                fileobj=docker_build_ctx,
                custom_context=True,
                encoding="gzip",
            )
        try:
            os.remove(build_ctx_path)
        except Exception:
            _logger.info("Temporary docker context file %s was not deleted.", build_ctx_path)
        tracking.MlflowClient().set_tag(self.run_id, MLFLOW_DOCKER_IMAGE_URI, image_uri)
        tracking.MlflowClient().set_tag(self.run_id, MLFLOW_DOCKER_IMAGE_ID, image.id)
        self._image = image

    def _get_docker_image_uri(self):
        """
        Returns an appropriate Docker image URI for a project based on the git hash of the specified
        working directory.

        :param repository_uri: The URI of the Docker repository with which to tag the image. The
                            repository URI is used as the prefix of the image URI.
        :param work_dir: Path to the working directory in which to search for a git commit hash
        """
        repository_uri = self._project.name
        repository_uri = repository_uri if repository_uri else "docker-project"
        # Optionally include first 7 digits of git SHA in tag name, if available.
        git_commit = _get_git_commit(self.work_dir)
        version_string = ":" + git_commit[:7] if git_commit else ""
        return repository_uri + version_string

    def _create_docker_build_ctx(self, dockerfile_contents):
        """
        Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
        """
        ignore_func = ignore_patterns(*get_paths_to_ignore(self.work_dir))
        # We are using the home directory of the user for the temporary directory
        # since otherwise the temporary directory will be created in the pwd
        # causing an infinite copy loop
        tmp_dir = root_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)
        directory = tempfile.mkdtemp(dir=tmp_dir)
        try:
            dst_path = os.path.join(directory, "mlflow-project-contents")
            shutil.copytree(src=self.work_dir, dst=dst_path, ignore=ignore_func)
            with open(os.path.join(dst_path, _GENERATED_DOCKERFILE_NAME), "w") as handle:
                handle.write(dockerfile_contents)
            _, result_path = tempfile.mkstemp(dir=tmp_dir)
            file_utils.make_tarfile(
                output_filename=result_path, source_dir=dst_path, archive_name=_PROJECT_TAR_ARCHIVE_NAME
            )
        finally:
            shutil.rmtree(directory)
        return result_path

    def get_command(self):
        docker_path = "docker"
        cmd = [docker_path, "run", "--rm"]

        docker_args = self._backend_config[PROJECT_DOCKER_ARGS]
        # TODO: this could be a shared function for containers
        cmd = convert_container_args_to_list(cmd, docker_args)

        env_vars = self.get_run_env_vars()
        tracking_cmds, tracking_envs = self._get_tracking_cmd_and_envs()
        artifact_cmds, artifact_envs = self._get_artifact_storage_cmd_and_envs()

        cmd += tracking_cmds + artifact_cmds
        env_vars.update(tracking_envs)
        env_vars.update(artifact_envs)
        user_env_vars = self._project.docker_env.get("environment")
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

        volumes = self._project.docker_env.get("volumes")
        if volumes is not None:
            for v in volumes:
                cmd += ["-v", make_volume_abs(v)]

        for key, value in env_vars.items():
            cmd += ["-e", "{key}={value}".format(key=key, value=value)]
        cmd += [self._image.tags[0]]
        return cmd

    def _get_tracking_cmd_and_envs(self):
        cmds = []
        env_vars = dict()

        tracking_uri = tracking.get_tracking_uri()
        local_path, container_tracking_uri = get_local_uri_or_none(tracking_uri)
        if local_path is not None:
            cmds = ["-v", "%s:%s" % (local_path, MLFLOW_CONTAINER_TRACKING_DIR_PATH)]
            env_vars[tracking._TRACKING_URI_ENV_VAR] = container_tracking_uri
        env_vars.update(get_databricks_env_vars(tracking_uri))
        return cmds, env_vars

    def _get_artifact_storage_cmd_and_envs(self):
        artifact_uri = self._active_run.info.artifact_uri
        artifact_repo = artifact_storage.get_artifact_repository(artifact_uri)
        _get_cmd_and_envs = artifact_storage._artifact_storages.get(type(artifact_repo))
        if _get_cmd_and_envs is not None:
            return _get_cmd_and_envs(artifact_repo)
        else:
            return [], {}

    def add_run_args(self, run_args):
        singularity_args = self._backend_config[PROJECT_DOCKER_ARGS]
        for key, value in singularity_args.items():
            args = key if isinstance(value, bool) else "%s=%s" % (key, value)
            run_args.extend(["--docker-args", args])
        return run_args

    @staticmethod
    def get_command_separator():
        return ' '