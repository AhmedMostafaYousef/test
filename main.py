from utils.allocateGPU import *

#allocate_gpu()

import _main
import parser_1

if __name__ == "__main__":
    print("cuda is available: ", torch.cuda.is_available())

    args = parser_1.parse_args()
    _main.main(args)
