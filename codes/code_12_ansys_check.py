import os
from ansys.aedt.core import Desktop

# CONFIGURATION: The verified path to your ANSYS Student installation
ansys_path = r"A:\AnsysEM\ANSYS Inc\ANSYS Student\v252\AnsysEM"
project_path = os.path.join(os.getcwd(), "Ansys_Connection_Test.aedt")

# --- PATH VERIFICATION ---
if not os.path.exists(ansys_path):
    print(f"❌ Error: The directory '{ansys_path}' does not exist.")
    print("Please double-check the folder name on your A: drive.")
    exit()

try:
    # Simplified call for the new ansys-aedt-core 0.27.1
    # We use student_version=True and the standard version string
    d = Desktop(version="2025.2", 
                non_graphical=True, 
                student_version=True)
    
    print("✅ Success! Python successfully launched ANSYS Electronics Desktop 2025 R2.")
    print(f"Current AEDT Version: {d.aedt_version_id}")
    
    # Close the session
    d.release_desktop()
    print("Session closed safely.")

except Exception as e:
    print(f"❌ Error: Could not connect to ANSYS at {ansys_path}")
    print(f"Details: {str(e)}")
    print("\nPlease verify that 'ansysedt.exe' exists in the path above.")
