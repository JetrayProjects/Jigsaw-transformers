# Modal Setup on Local Computer

## Create an account on Modal
1. [https://modal.com/]

2. In your terminal run

   `pip install modal`

   then

   `modal token new`
   This command would open the modal.com browser to login and after logging in create a API token that will be stored in your home directory.

4. Run Script

   I have written two scripts that run:
   - **sample_modal.py** : Runs **sample_fast.py**
   - **sample_modal_seed.py** : Runs **sample_from_seed**
  
   To run these files on the cloud service you would type:
   `modal run sample_modal.py` or `modal run sample_modal_seed.py` depending on your requirements.

   
