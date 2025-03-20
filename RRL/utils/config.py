import argparse
# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--data_name", type=str, default="wiki", # 
                    help="data name")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='noisylabel')
parser.add_argument('--ckpt_dir', type=str, default='noisylabel')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--ls', type=str, default='cos', help='lr scheduler')
parser.add_argument('--loss', type=str, default='GECE', help='CE RCE MAE') # MCE
parser.add_argument('--output_dim', type=int, default=512, help='output shape')
parser.add_argument('--noisy_ratio', type=float, default=0) 
parser.add_argument('--beta', type=float, default=0.8)
parser.add_argument('--tau', type=float, default=1.)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--views', nargs='+', help='<Required> Quantization bits', default=['Img', 'Txt', 'Audio', '3D', 'Video']) 
parser.add_argument('--choose_model',type=str,default='predata')#olddata ,mydata,predata
parser.add_argument('--low_dim', type=int, default=512)
parser.add_argument('--arch',type=str,default="resnet50",choices=["resnet50", "resnet18"])
parser.add_argument('--dset', type=str, default='a2r')
parser.add_argument('--dataset', default='office_home', choices=['office_home', 'office31', 'image_CLEF'])
parser.add_argument('--batch_size', type=int, default=16,help="batch_size")
parser.add_argument('--worker',type=int,default=4,help="number of workers")
parser.add_argument('--class_num',type=int,default=100,help="class_num")#根据数据集修改[100office_home,50office31,24image_CLEF]
parser.add_argument('--lambda_cross_domain',type=float,default=0.1,metavar='T',help="paramter for cross domain loss")
parser.add_argument('--experiment', default='true', choices=['xr', 'true'])
args = parser.parse_args()
print(args)
