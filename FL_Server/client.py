import io
BUFFER_SIZE = 1024

class Client():
    def __init__(self, clientId, conn, address):
        self.clientId = clientId
        self.conn = conn
        self.address = address
        #byte array of files
        self.actormodel = None
        self.criticmodel = None

    def getClientId(self):
        return self.clientId

    def getConn(self):
        return self.conn

    def getAddress(self):
        return self.address

    def saveActorModelFile(self, file):
        self.actormodel = file

    def getActorModelFile(self):
        return self.actormodel

    def saveCriticModelFile(self, file):
        self.criticmodel = file

    def getCriticModelFile(self):
        return self.criticmodel

    def transmitActorModel(self):
        print("Sending Actor Model Updates")
        self.conn.send(b'Actor_Model')
        try:
            for line in self.actormodel:
                self.conn.send(line)
            self.conn.send(b'END')

        except Exception as e:
            print(e)
            return False
        msg = self.conn.recv(BUFFER_SIZE)
        return True

    def transmitCriticModel(self):
        print("Sending Critic Model Updates")
        self.conn.send(b'Critic_Model')
        try:
            for line in self.criticmodel:
                self.conn.send(line)
            self.conn.send(b'END')

        except Exception as e:
            print(e)
            return False
        msg = print(self.conn.recv(BUFFER_SIZE))
        self.receiveActorModel()
        return True

    def receiveActorModel(self):
        recived_actor = 'jehotnahiyetechhhactor'+'.pth'
        recived_critic = 'jehotnahiyetechhhcritic'+'.pth'
        actormodelData = bytearray()
        criticmodelData = bytearray()
        self.conn.send(b'Send Model Updates')
        message = self.conn.recv(BUFFER_SIZE)
        print(message)

        if message == b'Sending_Actor_Model_updates':
            #Receive, output and save file
            with open(recived_actor, "wb") as fw:
                while True:
                    #print('receiving')
                    data = self.conn.recv(BUFFER_SIZE)
                    if data == b'ENDED' or data == b'':
                        print('Breaking from file write')
                        break
                    #print('Received: ', data.decode('utf-8'))
                    actormodelData.extend(data)
                    fw.write(data)
                    #print('Wrote to file', data.decode('utf-8'))
                fw.close()
            self.conn.send(b'Got the actor updates')

        message = self.conn.recv(BUFFER_SIZE)
        print(message)
        if message == b'Sending_Critic_Model_updates':
            #Receive, output and save file
            with open(recived_critic, "wb") as fw:
                while True:
                    #print('receiving')
                    data = self.conn.recv(BUFFER_SIZE)
                    if data == b'ENDED':
                        print('Breaking from file write')
                        break
                    #print('Received: ', data.decode('utf-8'))
                    fw.write(data)
                    #print('Wrote to file', data.decode('utf-8'))
                fw.close()
            self.conn.send(b'Got the critic updates')

        result = io.BytesIO(bytes(actormodelData))
        return result


