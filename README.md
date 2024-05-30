# MedEd AI
Create a RAG model using course content to generate a Google Forms quiz.
[Workflow in Excalidraw](https://excalidraw.com/#json=NpMYP2oIVYDmULCbdSe73,nX54wTxBQ98v32NCTNw9cg)

## 🚀 Getting Started
For getting started using this project, you will need:
* A `client_secrets.json` file or a `credentials.json` file using Google Cloud Console:
* An Azure OpenAI API Key. Once you have your API key, save it to `key.txt` in your project folder.
* A data folder, containing all the documents you'd like the RAG to use for context. These documents can be .pdf, .txt, or .doc files.

To get your `client_secrets.json` file or `credentials.json` file:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project if you do not have one already.
3. Go to `Enabled APIs & Services` using the menu on the left.
4. Click on `ENABLE APIs AND SERVICES`.
5. Type Google Forms API in the search box.
6. Click `ENABLE`.
7. Go to `CREDENTIALS` using the menu on the left.
8. Click on `CREATE CREDENTIALS`, and choose either `OAuth Client ID` or `Service Account`.
9. You may need to configure your application first.
10. Choose `Desktop Application` as your application type, then click `Create`.
11. Go back to `Credentials` using the menu on the left.
12. You will see your credentials have been created. Click the download button. (If you chose to create a `Service Account`, click into the service account name, then navigate to `Keys` and create your keys. Then download and save as `credentials.json`)
13. You may need to rename it and move it to your project folder.


## 🤔 How do I use it?
Using this model is surpisingly simple, thanks to Docker:
1. Run `docker build -t rag-demo .` (you can rename the image if you'd like).
2. Run `docker run -it --rm --name rag-demo rag-demo` (you can rename your container whatever you like.)
3. That's it! When the program is done, you will be given the link to your form.

What if I want to change something?
1. No problem. Once you clone this repo, you can change the prompt in `createQuiz.py`
2. Re-build and then re-run the container with the above steps.

## 🐛 Bugs & Issues
If you encounter any bugs, please [open up an issue](https://github.com/ccb-hms/MedEd-AI/issues).
