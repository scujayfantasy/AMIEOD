import torch
import torch.nn as nn

class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = nn.Conv2d(c1, c_, kernel_size=k, stride=s, padding=0, bias=True)                       #512 ---> 1280  (bs, 512, 20, 20) --->  (bs, 1280, 20, 20)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)                  #(bs, 1280, 20, 20) ---> (bs, 1280, 1, 1)
        self.drop = nn.Dropout(p=0.0, inplace=True)                            #p=0.0  没有丢弃任何神经元
        self.linear  = nn.Linear(c_, c2)  # to x(b,c2)                          #(bs, 1280, 1, 1) ---> (bs,2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):               #x:    (bs, 512, 20, 20)
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.softmax(self.linear(self.drop(self.pool(self.conv(x)).flatten(1))))
######################################################################################hxc

class En_Selecter_Cnn(nn.Module):
    def __init__(self, size = 256):
        super(En_Selecter_Cnn, self).__init__()
        self.size = size
        self.fc_in = int(self.size**2/32)
        channels = 16
        self.cnnnet = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True),   #/2
            nn.SiLU(),

            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=True),  #/2
            nn.SiLU(),

            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1, bias=True),  #/2
            nn.SiLU(),

            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1, bias=True),  #/2
            nn.SiLU(),

            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1, bias=True),  #/2
            nn.SiLU(),
        )

        self.classifier =  Classify(c1=32,c2=4)

        # self.full_layers1 = nn.Sequential(
        #     nn.Linear(self.fc_in, 64),
        #     # nn.Linear(2560, 64),
        #     nn.Linear(64, 15),
        # )

    def forward(self, x):

        x = nn.functional.interpolate(x, size=(self.size, self.size), mode = "bilinear", align_corners = False)

        out = self.cnnnet(x)  #  in (1,3,h,w)   out (1,32,h/32,w/32)
        # out = out.view(out.size(0), -1)     #  32 * h/32 * h/32
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    images = torch.randn(1, 3, 640, 640)

    selector = En_Selecter_Cnn(size = 256)

    out = selector(images)

    print(out.shape)


