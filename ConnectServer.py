import socket
import time

TCP_IP = 'localhost'
TCP_PORT = 9999
BUFFER_SIZE = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

torcs_server = int(input("Enter Torcs Server Port : "))

recieved_actor = 'received_globalmodel_actor'+str(torcs_server)+'.pth'
recieved_critic = 'received_globalmodel_critic'+str(torcs_server)+'.pth'
send_actor = 'Logs/Weights_Policy/policyweight_'+str(torcs_server)+'.pth'
send_critic = 'Logs/Weights_Critic/criticweight_'+str(torcs_server)+'.pth'
message_from_server = " "
message_to_server = ""
#loop infinite to keep client active
while True:
    message_from_server = s.recv(BUFFER_SIZE)
    print(message_from_server)
    if message_from_server == b'Actor_Model':
        #receive actor model
        with open(recieved_actor, 'wb') as f:
            print('file opened')
            while True:
                data = s.recv(BUFFER_SIZE)
                if data == b'END':
                    f.close()
                    print('file close()')
                    break
                # write data to a file
                f.write(data)
        print('Successfully got the file')
        message_to_server=b'Client Received_Actor_Model'
        s.send(b'Client Received_Actor_Model')
    elif message_from_server == b'Critic_Model':
        #receive critic model
        with open(recieved_critic, 'wb') as f:
            print('file opened')
            while True:
                data = s.recv(BUFFER_SIZE)
                if data == b'END':
                    f.close()
                    print('file close()')
                    break
                # write data to a file
                f.write(data)
        print('Successfully got the file')
        message_to_server=b'Client Received_Critic_Model'
        s.send(b'Client Received_Critic_Model')
    elif message_from_server == b'Send Model Updates' :
        print("Got innnnnnn")
        message_to_server="Sending_Actor_Model"
        s.send(b'Sending_Actor_Model_updates')

        with open(send_actor, 'rb') as fs:
            #Using with, no file close is necessary,
            #with automatically handles file close
            while True:
                data = fs.read(BUFFER_SIZE)
                #print('Sending data', data.decode('utf-8'))
                if not data:
                    print("end")
                    s.send(b'ENDED')
                    print('Breaking from sending data')
                    break
                s.send(data)
                print('Sent data', data)
        print(s.recv(BUFFER_SIZE))
        fs.close()
        print("reached")

        message_to_server="Sending_Critic_Model"
        s.send(b'Sending_Critic_Model_updates')
        with open(send_critic, 'rb') as fs:
            #Using with, no file close is necessary,
            #with automatically handles file close
            while True:
                data = fs.read(BUFFER_SIZE)
                #print('Sending data', data.decode('utf-8'))
                if not data:
                    print("end")
                    s.send(b'ENDED')
                    print('Breaking from sending data')
                    break
                s.send(data)
                #print('Sent data', data)
        fs.close()
        print("reached")
        s.recv(BUFFER_SIZE)





