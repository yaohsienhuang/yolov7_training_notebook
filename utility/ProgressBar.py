import time
from IPython.display import clear_output
from prettytable import PrettyTable

class progressBar():
    '''sample:
        progress_Bar=progressBar()
        progress_Bar.start(name='file move')
        progress_Bar.update(progress,total_progress)
    '''
    def start(self,name='progress'):
        self.start_time=time.time()
        self.name=name
        print('the {} starts at {} .'.format(self.name, time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(self.start_time))))
        time.sleep(5)
        
    def update(self,complete,total):
        update_time=time.time()
        spend_time=update_time - self.start_time
        progress=complete/total
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        bar_length = 20
        block = int(round(bar_length * progress))
        fps=float(complete/spend_time)
        clear_output(wait = True)
        text = "{}:{:.1f}% [{}] {}/{} {:.1f}s FPS={:.1f}".format(self.name,progress*100,"*"*block+"-"*(bar_length-block),complete,total,spend_time,fps)
        print(text)
        if progress ==1:
            self.end(total,spend_time,fps)
    
    def end(self,total,spend_time,fps):
        self.end_time=time.time()
        print('=======================================================')
        print('the {} ends at {} .'.format(self.name,time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.end_time))))
        my_table = PrettyTable()
        my_table.field_names = [ "Total", "Progress","Time(s)", "FPS"]
        my_table.add_row([ total,'100%',round(spend_time,1),round(fps,1)])
        print(my_table)
        