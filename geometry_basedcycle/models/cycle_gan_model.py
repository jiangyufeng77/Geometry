import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'D_B', 'G_B', 'G_A_cam', 'G_B_cam', 'cycle_A', 'cycle_B',
                           'recon_geo_A', 'recon_geo_B', 'recon_app_A', 'recon_app_B', 'identity_G_A', 'identity_G_B']  # 'reconwithin_G_A'
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_A_within', 'fake_A', 'recon_A']  # 'fake_A_t'
        visual_names_B = ['real_B', 'fake_B_within', 'fake_B', 'recon_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        # visual_names_A.append('idt_B')
        # visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # print(self.netG_A)

        # netG = networks.init_net(networks.EDGenerator, opt.init_type, opt.init_gain, self.gpu_ids)

        # netg = None
        # netg = networks.EDGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout)
        # self.netG_A = networks.init_net(netg, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.init_net(netg, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = networks.RhoClipper(0, 1)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        ## within domain
        self.appearance_within_A, self.geometry_within_A, _, _, gamma_within_A, beta_within_A = self.netG_A.encoder(self.real_A)
        self.appearance_within_B, self.geometry_within_B, _, _, gamma_within_B, beta_within_B = self.netG_B.encoder(self.real_B)
        self.fake_A_within = self.netG_A.decoder(self.appearance_within_A+self.geometry_within_A, gamma_within_A, beta_within_A)
        self.fake_B_within = self.netG_B.decoder(self.appearance_within_B+self.geometry_within_B, gamma_within_B, beta_within_B)
        # self.fake_A_within = self.netG_A(self.real_A)  # G_A(A)
        # self.fake_B_within = self.netG_B(self.real_B)  # G_B(B)

        ## cross domain
        self.appearance_A, self.geometry_A, cam_logit_A, heatmap_A, gamma_A, beta_A = self.netG_A.encoder(self.real_A)  # size(1, 256, 64, 64)
        self.appearance_B, self.geometry_B, cam_logit_B, heatmap_B, gamma_B, beta_B = self.netG_B.encoder(self.real_B)

        self.fake_B = self.netG_B.decoder(self.appearance_B+self.geometry_A, gamma_A, beta_A)
        self.fake_A = self.netG_A.decoder(self.appearance_A+self.geometry_B, gamma_B, beta_B)

        self.appearance_B_Rec, self.geometry_A_Rec, _, _, gamma_A_Rec, beta_A_Rec = self.netG_B.encoder(self.fake_B)
        self.appearance_A_Rec, self.geometry_B_Rec, _, _, gamma_B_Rec, beta_B_Rec = self.netG_A.encoder(self.fake_A)

        self.recon_B = self.netG_B.decoder(self.geometry_B_Rec+self.appearance_B_Rec, gamma_B_Rec, beta_B_Rec)
        self.recon_A = self.netG_A.decoder(self.geometry_A_Rec+self.appearance_A_Rec, gamma_A_Rec, beta_A_Rec)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, pred_real_cam, _ = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_real_cam = self.criterionGAN(pred_real_cam, True)
        # Fake
        pred_fake, pred_fake_cam, _ = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_fake_cam = self.criterionGAN(pred_fake_cam, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake + loss_D_real_cam + loss_D_fake_cam) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        identity_weight = 10

        # within domain
        # self.loss_within_G_A = self.criterionGAN(self.netD_A(self.fake_A_within), True)
        self.loss_identity_G_A = self.criterionCycle(self.fake_A_within, self.real_A)
        # self.loss_within_G_B = self.criterionGAN(self.netD_B(self.fake_B_within), True)
        self.loss_identity_G_B = self.criterionCycle(self.fake_B_within, self.real_B)

        # cross domain
        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A), True)
        self.pred_G_A, self.pred_G_A_cam, _ = self.netD_A(self.fake_A)
        self.loss_G_A = self.criterionGAN(self.pred_G_A, True)
        self.loss_G_A_cam = self.criterionGAN(self.pred_G_A_cam, True)
        # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.pred_G_B, self.pred_G_B_cam, _ = self.netD_B(self.fake_B)
        self.loss_G_B = self.criterionGAN(self.pred_G_B, True)
        self.loss_G_B_cam = self.criterionGAN(self.pred_G_B_cam, True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.recon_A, self.real_A)
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.recon_B, self.real_B) #  * lambda_B
        # combined loss and calculate gradients
        self.loss_recon_geo_A = self.criterionCycle(self.geometry_A_Rec, self.geometry_A)
        self.loss_recon_geo_B = self.criterionCycle(self.geometry_B_Rec, self.geometry_B) # * lambda_B
        self.loss_recon_app_A = self.criterionCycle(self.appearance_A_Rec, self.appearance_A)
        self.loss_recon_app_B = self.criterionCycle(self.appearance_B_Rec, self.appearance_B) # * lambda_B
        # self.loss_G = self.loss_reconwithin_G_A + self.loss_reconwithin_G_B + self.loss_recon_geo_A + self.loss_recon_geo_B + self.loss_recon_app_A + self.loss_recon_app_B + \
        #               self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B  # self.loss_reconwithin_G_A B

        self.loss_G = self.loss_G_A + self.loss_G_A_cam + self.loss_G_B + self.loss_G_B_cam + \
                      identity_weight * (self.loss_identity_G_A + self.loss_identity_G_B) + \
                      lambda_A * (self.loss_cycle_A + self.loss_recon_geo_A + self.loss_recon_app_A) + \
                      lambda_B * (self.loss_cycle_B + self.loss_recon_geo_B + self.loss_recon_app_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.netG_A.apply(self.Rho_clipper)
        self.netG_B.apply(self.Rho_clipper)

