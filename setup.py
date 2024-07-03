import subprocess
import os

class Setup:
    """
    This class defines the setup process to check and install required packages.
    
    Attributes
    ----------
    requirements_dir:
        The directory containing the package requirements.
        
    requirements_file:
        The requirements file containing the package names.
        
    install_script:
        The script to install the packages.
        
    setup_flag:
        The flag to indicate that the setup has been completed.
        
    Methods
    -------
    check_and_install_packages:
        Check and install required packages.
        
    print_current_directory:
        Print the current working directory.
        
    run_install_script:
        Run the install script to install packages.
        
    requirements_file_exists:
        Check if the requirements file exists.
        
    setup_completed:
        Check if the setup has been completed.

    create_setup_flag:
        Create the setup flag.
    """
    def __init__(self):
        self.requirements_dir = 'package_requirements'
        self.requirements_file = os.path.join(self.requirements_dir, 'requirements.txt')
        self.install_script = os.path.join(self.requirements_dir, 'install_requirements.sh')
        self.setup_flag = 'setup_completed.flag'
        self.check_and_install_packages()

    def check_and_install_packages(self):
        """
        Check and install required packages.
        """
        
        if not self.setup_completed():
            print("Checking and installing required packages...")
            self.print_current_directory()
            self.run_install_script()
            self.create_setup_flag()
        else:
            print("Setup has already been completed. Skipping package installation.")

    def print_current_directory(self):
        """
        Print the current working directory.
        """
        print(f"Current working directory: {os.getcwd()}")

    def run_install_script(self):
        """
        Run the install script to install packages.
        
        Raises
        ------
        ValueError:
            If an error occurs while installing the packages.
        """
        try:
            if self.requirements_file_exists():
                print(f"Changing directory to {self.requirements_dir}...")
                os.chdir(self.requirements_dir)
                print(f"Running {self.install_script} to install packages...")
                subprocess.check_call(['bash', 'install_requirements.sh'])
                print("Packages installed successfully.")
                os.chdir('..')
            else:
                print(f"{self.requirements_file} not found. No packages to install.")
        except subprocess.CalledProcessError as e:
            raise ValueError("An error occurred while installing the packages.") from e

    def requirements_file_exists(self):
        return os.path.exists(self.requirements_file)

    def setup_completed(self):
        return os.path.exists(self.setup_flag)

    def create_setup_flag(self):
        with open(self.setup_flag, 'w') as f:
            f.write('Setup completed')

if __name__ == "__main__":
    setup = Setup()
