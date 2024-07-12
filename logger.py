
"""

"""
import os
import datetime
# logger = Logger(args, epoch, train_acc1, train_acc2, val_acc1, val_acc2, None, train_loss, val_loss, None, None, 'SSPL')
class Logger():
    def __init__(self, args, epoch, t_acc1, t_acc2, v_video_acc1, v_video_acc2, dual_v_video_acc, t_loss, v_video_loss, dual_v_video_loss, info, mode):
        self.args = args
        self.epoch = epoch
        self.t_acc1 = t_acc1
        self.v_video_acc1 = v_video_acc1
        self.t_acc2 = t_acc2
        self.v_video_acc2 = v_video_acc2
        self.dual_v_video_acc = dual_v_video_acc
        self.t_loss = t_loss
        self.v_video_loss = v_video_loss
        self.dual_v_video_loss = dual_v_video_loss
        self.info = info
        self.mode = mode
        
        
    
    def acc_log(self):
        if self.mode != 'SSPL':
            self.acc_logger1(self.args, self.epoch, self.t_acc1, self.v_video_acc1,  self.dual_v_video_acc)
        else:
            self.acc_logger2(self.args, self.epoch, self.t_acc1, self.t_acc2, self.v_video_acc1, self.v_video_acc2, self.dual_v_video_acc)
            
    def loss_log (self):
        if self.mode != 'SSPL':
            self.loss_logger1(self.args, self.epoch, self.t_loss, self.v_video_loss, self.dual_v_video_loss )
        else:
            self.loss_logger2(self.args, self.epoch, self.t_loss, self.v_video_loss, self.dual_v_video_loss )
            
    def deep_test_log (self):
        self.deep_test_logger(self.args, self.info)
        
        
        
    def acc_logger1(self, args, epoch, t_acc, v_video_acc, dual_v_video_acc):
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%Y %H:%M:%S")
        
        # Writing to a file
        log_acc_path = args.log+'\\'+'acc_log.txt'
        if (os.path.exists(log_acc_path)):
            with open(log_acc_path, "a") as file:
            
                file.write((f'{now_str:20} {epoch:03} {t_acc:.20f} {v_video_acc:.20f} {dual_v_video_acc:.20f} \n'))
        else:
            with open(log_acc_path, "w+") as file:
           
                file.write(( 'Date           Epoch        Train Acc     Val Video Acc  All Clips Video Acc \n'))
                file.write((f'{now_str:20} {epoch:03} {t_acc:.20f} {v_video_acc:.20f} {dual_v_video_acc:.20f} \n'))
                
                
    def acc_logger2(self, args, epoch, t_acc1, t_acc2, v_video_acc1, v_video_acc2, dual_v_video_acc):
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%Y %H:%M:%S")
        
        # Writing to a file
        log_acc_path = args.log+'\\'+'acc_log.txt'
        if (os.path.exists(log_acc_path)):
            with open(log_acc_path, "a") as file:
            
                file.write((f'{now_str:20} {epoch:03} {t_acc1:.20f} {t_acc2:.20f} {v_video_acc1:.20f} {v_video_acc2:.20f} \n'))
        else:
            with open(log_acc_path, "w+") as file:
           
                file.write(( 'Date           Epoch      Train Acc1    Train Acc2   Val Video Acc1       Val Video Acc2 \n'))
                file.write((f'{now_str:20} {epoch:03} {t_acc1:.20f} {t_acc2:.20f} {v_video_acc1:.20f} {v_video_acc2:.20f} \n'))




    def loss_logger1(self, args, epoch, t_loss, v_video_loss, dual_v_video_loss):
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%Y %H:%M:%S")
        
        # Writing to a file
        log_loss_path=args.log+'\\'+'loss_log.txt'
        if (os.path.exists(log_loss_path)):
            with open(log_loss_path, "a") as file:
            
                file.write((f'{now_str:20} {epoch:03} {t_loss:.20f} {v_video_loss:.20f} {dual_v_video_loss:.20f} \n'))
        else:
            with open(log_loss_path, "w+") as file:
           
                file.write(( 'Date          Epoch       Train loss       Val Video Loss All Clips Video Loss \n'))
                file.write((f'{now_str:20} {epoch:03} {t_loss:.20f} {v_video_loss:.20f} {dual_v_video_loss:.20f} \n'))
                
         
    def loss_logger2(self, args, epoch, t_loss, v_video_loss, dual_v_video_loss):
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%Y %H:%M:%S")
        
        # Writing to a file
        log_loss_path=args.log+'\\'+'loss_log.txt'
        if (os.path.exists(log_loss_path)):
            with open(log_loss_path, "a") as file:
            
                file.write((f'{now_str:20} {epoch:03} {t_loss:.20f} {v_video_loss:.20f} \n'))
        else:
            with open(log_loss_path, "w+") as file:
           
                file.write(( 'Date          Epoch       Train loss       Val Video Loss \n'))
                file.write((f'{now_str:20} {epoch:03} {t_loss:.20f} {v_video_loss:.20f} \n'))
  
            
    def deep_test_logger(self, args, info):
        now = datetime.datetime.now()
        now_str = now.strftime("%d-%m-%Y %H:%M:%S")
    
        # Writing to a file
        deep_test_path = args.log+'\\'+'deep_test.txt'
        if (os.path.exists(deep_test_path)):
            with open(deep_test_path, "a") as file:
            
                file.write((f'{info} \n'))
        else:
            with open(deep_test_path, "w+") as file:
           
                #file.write(( ' Epoch       Train loss       Val Video Loss \n'))
                file.write((f'{info} \n'))
            
  