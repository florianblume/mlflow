import logging
import os
import subprocess
import sys

import mlflow

from mlflow.projects import (
    RunEnvironment,
    DockerRunEnvironment,
    SingularityRunEnvironment,
    CondaRunEnvironment
)

from mlflow.projects.submitted_run import LocalSubmittedRun
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.utils import (
    fetch_and_validate_project,
    get_or_create_run,
    load_project,
    get_databricks_env_vars,
    get_entry_point_command,
    MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG,
    PROJECT_USE_CONDA,
    PROJECT_SYNCHRONOUS,
    PROJECT_STORAGE_DIR,
)
from mlflow.utils.conda import get_conda_command, get_or_create_conda_env
from mlflow import tracking
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV


_logger = logging.getLogger(__name__)


RUN_ENVIRONMENTS = {
    'docker': DockerRunEnvironment,
    'singularity': SingularityRunEnvironment,
    'conda': CondaRunEnvironment
}


class LocalBackend(AbstractBackend):
    def run(
        self, project_uri, entry_point, params, version, backend_config, tracking_uri, experiment_id
    ):
        # Validates that there is only one environment defined in the project
        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        project = load_project(work_dir)

        if MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG in backend_config:
            run_id = backend_config[MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG]
        else:
            run_id = None
        active_run = get_or_create_run(
            run_id, project_uri, experiment_id, work_dir, version, entry_point, params
        )

        command_args = []
        command_separator = ' '

        env_name = ''
        run_environment = None
        if backend_config[PROJECT_USE_CONDA]:
            env_name = 'conda'
            run_environment = RUN_ENVIRONMENTS['conda']
        if project.docker_env:
            env_name = 'docker'
            run_environment = RUN_ENVIRONMENTS['docker']
        if project.singularity_env:
            env_name = 'singularity'
            run_environment = RUN_ENVIRONMENTS['singularity']
        else:
            # The user wants to run their code in a system environment and doesn't want to activate anything
            pass

        if run_environment is not None:
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, env_name)
            run_environment = run_environment(work_dir, project, active_run, backend_config)
            run_environment.validate_installation()
            run_environment.validate_environment()
            run_environment.prepare_environment()
            command_args += run_environment.get_command()
            command_separator = run_environment.get_command_separator()

        # In synchronous mode, run the entry point command in a blocking fashion, sending status
        # updates to the tracking server when finished. Note that the run state may not be
        # persisted to the tracking server if interrupted
        storage_dir = backend_config[PROJECT_STORAGE_DIR]
        synchronous = backend_config[PROJECT_SYNCHRONOUS]
        if synchronous:
            command_args += get_entry_point_command(project, entry_point, params, storage_dir)
            command_str = command_separator.join(command_args)
            return _run_entry_point(
                command_str, run_environment
            )
        use_conda = backend_config[PROJECT_USE_CONDA]
        # Otherwise, invoke `mlflow run` in a subprocess
        return _invoke_mlflow_run_subprocess(
            work_dir=work_dir,
            entry_point=entry_point,
            parameters=params,
            experiment_id=experiment_id,
            run_environment=run_environment,
            use_conda=use_conda,
            storage_dir=storage_dir
        )


def _invoke_mlflow_run_subprocess(
    work_dir, entry_point, parameters, run_environment, use_conda, storage_dir
):
    """
    Run an MLflow project asynchronously by invoking ``mlflow run`` in a subprocess, returning
    a SubmittedRun that can be used to query run status.
    """
    _logger.info("=== Asynchronously launching MLflow run with ID %s ===", run_environment.run_id)
    mlflow_run_arr = _build_mlflow_run_cmd(
        uri=work_dir,
        entry_point=entry_point,
        run_environment=run_environment,
        storage_dir=storage_dir,
        run_id=run_environment.run_id,
        parameters=parameters,
    )
    env_vars = run_environment.get_run_env_vars()
    env_vars.update(get_databricks_env_vars(mlflow.get_tracking_uri()))
    mlflow_run_subprocess = _run_mlflow_run_cmd(mlflow_run_arr, env_vars)
    return LocalSubmittedRun(run_environment.run_id, mlflow_run_subprocess)


def _build_mlflow_run_cmd(
    uri, entry_point, run_environment, use_conda, storage_dir, run_id, parameters
):
    """
    Build and return an array containing an ``mlflow run`` command that can be invoked to locally
    run the project at the specified URI.
    """
    mlflow_run_arr = ["mlflow", "run", uri, "-e", entry_point, "--run-id", run_id]
    if run_environment is not None:
        mlflow_run_arr = run_environment.add_run_args(mlflow_run_arr)
    if storage_dir is not None:
        mlflow_run_arr.extend(["--storage-dir", storage_dir])
    if not use_conda:
        mlflow_run_arr.append("--no-conda")
    for key, value in parameters.items():
        mlflow_run_arr.extend(["-P", "%s=%s" % (key, value)])
    return mlflow_run_arr


def _run_mlflow_run_cmd(mlflow_run_arr, env_map):
    """
    Invoke ``mlflow run`` in a subprocess, which in turn runs the entry point in a child process.
    Returns a handle to the subprocess. Popen launched to invoke ``mlflow run``.
    """
    final_env = os.environ.copy()
    final_env.update(env_map)
    # Launch `mlflow run` command as the leader of its own process group so that we can do a
    # best-effort cleanup of all its descendant processes if needed
    if sys.platform == "win32":
        return subprocess.Popen(
            mlflow_run_arr,
            env=final_env,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        return subprocess.Popen(
            mlflow_run_arr, env=final_env, universal_newlines=True, preexec_fn=os.setsid
        )


def _run_entry_point(command, run_environment):
    """
    Run an entry point command in a subprocess, returning a SubmittedRun that can be used to
    query the run's status.
    :param command: Entry point command to run
    :param work_dir: Working directory in which to run the command
    :param run_id: MLflow run ID associated with the entry point execution.
    """
    env = os.environ.copy()
    env.update(run_environment.get_run_env_vars())
    env.update(get_databricks_env_vars(tracking_uri=mlflow.get_tracking_uri()))
    _logger.info("=== Running command '%s' in run with ID '%s' === ", command, run_environment.run_id)
    # in case os name is not 'nt', we are not running on windows. It introduces
    # bash command otherwise.
    if os.name != "nt":
        process = subprocess.Popen(["bash", "-c", command], close_fds=True, cwd=run_environment.work_dir, env=env)
    else:
        # process = subprocess.Popen(command, close_fds=True, cwd=work_dir, env=env)
        process = subprocess.Popen(["cmd", "/c", command], close_fds=True, cwd=work_dir, env=env)
    return LocalSubmittedRun(run_environment.run_id, process)
