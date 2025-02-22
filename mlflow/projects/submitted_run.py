from abc import abstractmethod

import os
import signal
import logging
import subprocess

from mlflow.entities import RunStatus

_logger = logging.getLogger(__name__)


class SubmittedRun(object):
    """
    Wrapper around an MLflow project run (e.g. a subprocess running an entry point
    command or a Databricks job run) and exposing methods for waiting on and cancelling the run.
    This class defines the interface that the MLflow project runner uses to manage the lifecycle
    of runs launched in different environments (e.g. runs launched locally or on Databricks).

    ``SubmittedRun`` is not thread-safe. That is, concurrent calls to wait() / cancel()
    from multiple threads may inadvertently kill resources (e.g. local processes) unrelated to the
    run.

    NOTE:

        Subclasses of ``SubmittedRun`` must expose a ``run_id`` member containing the
        run's MLflow run ID.
    """

    @abstractmethod
    def wait(self):
        """
        Wait for the run to finish, returning True if the run succeeded and false otherwise. Note
        that in some cases (e.g. remote execution on Databricks), we may wait until the remote job
        completes rather than until the MLflow run completes.
        """
        pass

    @abstractmethod
    def get_status(self):
        """
        Get status of the run.
        """
        pass

    @abstractmethod
    def cancel(self):
        """
        Cancel the run (interrupts the command subprocess, cancels the Databricks run, etc) and
        waits for it to terminate. The MLflow run status may not be set correctly
        upon run cancellation.
        """
        pass

    @property
    @abstractmethod
    def run_id(self):
        pass


class LocalSubmittedRun(SubmittedRun):
    """
    Instance of ``SubmittedRun`` corresponding to a subprocess launched to run an entry point
    command locally.
    """

    def __init__(self,  command_proc):
        super().__init__()
        self.command_proc = command_proc

    def wait(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        # Interrupt child process if it hasn't already exited
        if self.command_proc.poll() is None:
            # Kill the the process tree rooted at the child if it's the leader of its own process
            # group, otherwise just kill the child
            try:
                if self.command_proc.pid == os.getpgid(self.command_proc.pid):
                    os.killpg(self.command_proc.pid, signal.SIGKILL)
                else:
                    self.command_proc.kill()
            except OSError:
                # The child process may have exited before we attempted to terminate it, so we
                # ignore OSErrors raised during child process termination
                _logger.info(
                    "Failed to terminate child process (PID %s) corresponding to MLflow The process may have already exited.",
                    self.command_proc.pid,
                )
            self.command_proc.wait()

    def _get_status(self):
        exit_code = self.command_proc.poll()
        if exit_code is None:
            return RunStatus.RUNNING
        if exit_code == 0:
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self):
        return RunStatus.to_string(self._get_status())
