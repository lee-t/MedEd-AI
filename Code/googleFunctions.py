from googleapiclient.discovery import build
from apiclient import discovery
from httplib2 import Http
from googleapiclient import discovery
from google.oauth2 import service_account
from oauth2client import client, file, tools
from google_auth_oauthlib.flow import InstalledAppFlow
import json

class Form:

    def __init__(self, file_type: str, loginfile: str, discovery_doc: str, scopes: list, sentence_response):
        
        if file_type == 'client_secrets':
            flow = InstalledAppFlow. \
                from_client_secrets_file(loginfile, scopes=scopes)
            self.credentials = flow.run_local_server(port=0)

        elif file_type == 'credentials':
            self.credentials = service_account.Credentials. \
                from_service_account_file(loginfile, scopes=scopes)

        else:
            raise ValueError('Invalid file type for credentials.')

        self.form_service = discovery.build('forms', 'v1',
                                            credentials=self.credentials,
                                            discoveryServiceUrl=discovery_doc,
                                            static_discovery=False)
        
        self.del_requests =[]
        self.requests = []

        self.create_quiz(sentence_response)

    def add_question(self, question, index,
                        modify=False,
                        qtype="RADIO",
                        choices=[],
                        required=True
                        ):
        request = {
            "createItem": {
                "item": {
                    "title": question,
                    "questionItem": {
                        "question": {
                            "required": required,
                            "choiceQuestion": {
                                "type": qtype,
                                "options": self.create_choices(choices)
                            }
                        }
                    }
                },
                "location": {
                    "index": index
                }
            }
        }
        if modify:
            self.del_question(index)
        self.requests.append(request)

    def modify_form(self, title: str = None, description: str = None):
        request = {
            "updateFormInfo": {
                "info": {},
                "updateMask": ""
            }
        }
        if title:
            request['updateFormInfo']['info']['title'] = title
            request['updateFormInfo']['updateMask'] += 'title'
        if description:
            request['updateFormInfo']['info']['description'] = description
            request['updateFormInfo']['updateMask'] += ',description'

        self.requests.append(request)

    def del_question(self, index):
        request = {
            "deleteItem": {
                "location": {
                    "index": index
                }
            }
        }
        self.del_requests.append(request)

    def create_choices(self, choices):
        nchoices = []
        for choice in choices:
            nchoices.append({'value': choice})

        return nchoices

    def create_form(self, title):
        title = title if title else 'Untitled Form'
        form = {
            "info": {
                "title": title,
            }
        }
        result = self.form_service.forms().create(body=form).execute()
        self.form_id = result['formId']
        return self.form_id

    def get_form(self, form_id):
        return self.form_service.forms().get(formId=self.form_id).execute()


    def request_updates(self, form_id):
        self.del_requests.sort(
            key=lambda x: x['deleteItem']['location']['index'],
            reverse=True)

        if self.del_requests:
            self.form_service.forms(). \
                batchUpdate(formId=self.form_id,
                            body={'requests': self.del_requests}).execute()
        if self.requests:
            self.form_service.forms(). \
                batchUpdate(formId=self.form_id,
                            body={'requests': self.requests}).execute()

    def get_link_to_form(form_id):
        return "https://docs.google.com/forms/d/{}".format(form_id)

    
    def create_quiz(self, sentence_response):
            
        quiz = json.loads(sentence_response.response)

        self.form_id = self.create_form(title=quiz['info']['title'])
        self.modify_form(title=quiz['info']['title'], description=quiz['info']['description'])
        self.request_updates(self.form_id)

        for question in quiz['info']['questions']:
            self.add_question(question=question['question'],
                            index=quiz['info']['questions'].index(question),
                            qtype=question['type'],
                            choices=question['options'])

        self.request_updates(self.form_id)
        
        link=self.get_link_to_form(self.form_id)
        print(link)

    # def get_form_responses(self, form_id):
        
    #     result = self.form_service.forms().responses().list(formId=form_id).execute()
    #     return result