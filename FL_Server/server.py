
import copy
import gc
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import time

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from FL_Server.models import *
from FL_Server.utils import *
from FL_Server.Connection import *


logger = logging.getLogger(__name__)

BUFFER_SIZE = 1024


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning

    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.
    """

    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, conn_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model_config = model_config
        self.actormodel = eval(model_config["actor"]["name"])(
            **model_config["actor"])

        self.criticmodel = eval(model_config["critic"]["name"])(
            **model_config["critic"])

        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config

        self.actormodelPath = fed_config["actormodel_path"]
        self.criticmodelPath = fed_config["criticmodel_path"]

        self.conn_config = conn_config

        self.conn = startTCPServer()
        # time.sleep(12)
        # self.conn = ServerConn(self.conn_config)
        # self.mqtt_client = self.conn.connect_mqtt()
        # self.conn = ServerConn(self.conn_config)
        # self.conn.connect()
        # t1 = threading.Thread(target=self.conn.subscribing, name='t1')
        # t1.start()

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""

        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.actormodel, **self.init_config)
        init_net(self.criticmodel, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized actor model (# parameters: {str(sum(p.numel() for p in self.actormodel.parameters()))})!"
        self.logMessage(message)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized critic model (# parameters: {str(sum(p.numel() for p in self.criticmodel.parameters()))})!"
        self.logMessage(message)

        forward = input(" enter any no to proceed : ")
        clientsConnections = getAllClients()
        self.clients = (clientsConnections)
        # send the model skeleton to all clients
        if(len(clientsConnections)>=3):
            self.transmit_model()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        self.actormodel.load_state_dict(torch.load(self.actormodelPath,map_location=torch.device('cpu')))
        self.criticmodel.load_state_dict(torch.load(self.criticmodelPath,map_location=torch.device('cpu')))

        actormodelFile = open(self.actormodelPath, 'rb')
        criticmodelFile = open(self.criticmodelPath, 'rb')
        actormodelFileByteArray = self.convertFileToByteArray(actormodelFile)
        criticmodelFileByteArray = self.convertFileToByteArray(criticmodelFile)
        clientIndices = []

        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)
            for client in tqdm(self.clients, leave=False):
                client.saveActorModelFile(copy.deepcopy(
                    actormodelFileByteArray))
                client.saveCriticModelFile(copy.deepcopy(
                    criticmodelFileByteArray))

            for i in range(0, len(self.clients)):
                clientIndices.append(i)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully copied models to all {str(len(self.clients))} client instances!"
            self.logMessage(message)

        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in sampled_client_indices:
                print("index---------------------" + str(idx))
                print("length of clients: "+str(len(self.clients)))
                self.clients[idx].saveActorModelFile(copy.deepcopy(
                    actormodelFileByteArray))
                self.clients[idx].saveCriticModelFile(copy.deepcopy(
                    criticmodelFileByteArray))

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully copied models to selected {str(self.num_clients)} client instances!"
            self.logMessage(message)
            clientIndices = sampled_client_indices

        """Multiprocessing-applied version of "update_selected_clients" method."""
        with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
            workhorse.map(self.send_model_to_clients, clientIndices)
        # print(result)

    def collectModelsFromSelectedClients(self, sampled_client_indices):
        with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
            models = workhorse.map(
                self.collectModelClient, sampled_client_indices)
        actormodels = []
        criticmodels = []
        for model in models:
            actormodels.append(model[0])
            criticmodels.append(model[1])

        return actormodels, criticmodels


    def collectModelClient(self, selectedClientInd):
        # transmit actor model to a client
        client_id = self.clients[selectedClientInd].getClientId()
        result,result2 = self.clients[selectedClientInd].receiveActorModel()
        clientActorModel = None
        clientCriticModel = None
        if(result is not None):
            print("---------------------------------result not none")
            clientActorModel = eval(self.model_config["actor"]["name"])(
                **self.model_config["actor"])
            init_net(clientActorModel, **self.init_config)
            state_dict = torch.load(result)
            clientActorModel.load_state_dict(state_dict)
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} successfully received actor model"
            return clientActorModel
        else:
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} error occured while receiving actor model"
        self.logMessage(message)

        if(result2 is not None):
            clientCriticModel = eval(self.model_config["critic"]["name"])(
                **self.model_config["critic"])
            init_net(clientActorModel, **self.init_config)
            state_dict = torch.load(result)
            clientCriticModel.load_state_dict(state_dict)
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} successfully received critic model"
        else:
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} error occured while receiving critic model"

        self.logMessage(message)

        return clientActorModel, clientCriticModel

        # # transmit critic model to a client
        # result = self.clients[selectedClientInd].receiveCriticModel()
        # if(result):
        #     message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} successfully received critic model"
        # else:
        #     message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} error occured while receiving critic model"
        # self.logMessage(message)

    def send_model_to_clients(self, selectedClientInd):
        # update selected clients
        client_id = str(self.clients[selectedClientInd].getClientId())
        message = f"[Round: {str(self._round).zfill(4)}] Start sending file to the selected client {client_id.zfill(4)}...!"
        self.logMessage(message)

        # transmit actor model to a client
        result = self.clients[selectedClientInd].transmitActorModel()
        if(result):
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} file successfully transmitted actor model"
        else:
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} error occured while transmitting actor model"
        self.logMessage(message)

        # transmit critic model to a client
        result = self.clients[selectedClientInd].transmitCriticModel()
        if(result):
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} file successfully transmitted critic model"
        else:
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} error occured while transmitting critic model"
        self.logMessage(message)

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"reward": [], "damage": [], "steps": []}
        for r in range(self.num_rounds):
            self._round = r + 1

            self.train_federated_model()
            # test_loss, test_accuracy = self.evaluate_global_model()
            #
            # self.results['loss'].append(test_loss)
            # self.results['accuracy'].append(test_accuracy)
            #
            # self.writer.add_scalars(
            #     'Loss',
            #     {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
            #     self._round
            #     )
            # self.writer.add_scalars(
            #     'Accuracy',
            #     {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
            #     self._round
            #     )
            #
            # message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
            #     \n\t[Server] ...finished evaluation!\
            #     \n\t=> Loss: {test_loss:.4f}\
            #     \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"
            # print(message)
            # logging.info(message)
            # del message
            # gc.collect()
        #self.transmit_model()

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices =[0]# self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # list of collected models
        actorModels,criticmodels = self.collectModelsFromSelectedClients(
            sampled_client_indices)

        coefficient = 1.0 / len(self.clients)
        self.saveActorModel(self.average_model(actorModels, coefficient))
        self.saveCriticModel(self.average_model(criticModels, coefficient))

    def average_model(self, models, coefficient):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        it = 0
        #print(models)
        for model in models:
            # get model updates here
            local_weights = model.state_dict()

            for key in model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficient * local_weights[key]
                else:
                    averaged_weights[key] += coefficient * local_weights[key]

        return averaged_weights

    def saveActorModel(self, averaged_weights):
        self.actormodel.load_state_dict(averaged_weights)
        # save model
        torch.save(self.actormodel.state_dict(),
                   "./models/aggregated_actor.pth")

    def saveCriticModel(self, averaged_weights):
        self.criticmodel.load_state_dict(averaged_weights)
        # save model
        torch.save(self.criticmodel.state_dict(),
                   "./models/aggregated_critic.pth")

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        self.logMessage(message)

        clientsConnected = len(self.clients)
        num_sampled_clients = max(int(self.fraction * clientsConnected), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(
            self.num_clients-1)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def logMessage(self, message):
        print(message)
        logging.info(message)
        del message
        gc.collect()

    def convertFileToByteArray(self, file):
        byteArrOfFile = []
        while True:
            line = file.read(BUFFER_SIZE)
            while (line):
                byteArrOfFile.append(line)
                line = file.read(BUFFER_SIZE)
            if not line:
                file.close()
                break
        return byteArrOfFile
