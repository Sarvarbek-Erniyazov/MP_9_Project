import os
import logging
from datetime import datetime

class CustomLogger:
    def __init__(self):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.log_dir, self.log_file)
        
        logging.basicConfig(
            filename=self.log_file_path,
            format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        
        self.logger = logging.getLogger("EnergyForecasting")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[ %(levelname)s ] - %(message)s"))
        self.logger.addHandler(console_handler)

logging_instance = CustomLogger().logger