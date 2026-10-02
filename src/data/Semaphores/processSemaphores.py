# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.data.Semaphores.threads.threadSemaphores import threadSemaphores
try:
    from src.utils.logConfig import get_logger
except ModuleNotFoundError:
    class _PrintLogger:
        def __getattr__(self, _):
            return print
    def get_logger(name):
        return _PrintLogger()

class processSemaphores(WorkerProcess):
    """This process will receive the location of the other cars and the location and the state of the semaphores.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, ready_event=None, debugging = False):
        self.queuesList = queueList
        self.debugging = debugging
        super(processSemaphores, self).__init__(self.queuesList, ready_event)


    # ===================================== INIT TH ======================================
    def _init_threads(self):
        """Create the thread and add to the list of threads."""

        CarsSemTh = threadSemaphores(self.queuesList, self.debugging)
        self.threads.append(CarsSemTh)


# =================================== EXAMPLE =========================================

if __name__ == "__main__":
    from multiprocessing import Queue

    queueList = {
        "Critical": Queue(),  # Queue for critical messages
        "Warning": Queue(),  # Queue for warning messages
        "General": Queue(),  # Queue for general messages
        "Config": Queue(),  # Queue for configuration messages
    }

    allProcesses = list()
    process = processSemaphores(queueList)
    process.start()

    x = range(6)
    for n in x:
        get_logger("Semaphores").info(queueList["General"].get())  # Print general messages

    process.stop()