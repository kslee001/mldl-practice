import torch
import torch.nn.functional as F
from math import sqrt

# private
import detection_utils




class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x))


class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            Conv(3, 64, kernel_size=3, padding=1, stride=1),
            Conv(64, 64, kernel_size=3, padding=1, stride=1))
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Sequential(
            Conv(64, 128, kernel_size=3, padding=1, stride=1),
            Conv(128, 128, kernel_size=3, padding=1, stride=1))
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Sequential(
            Conv(128, 256, kernel_size=3, padding=1, stride=1),
            Conv(256, 256, kernel_size=3, padding=1, stride=1),
            Conv(256, 256, kernel_size=3, padding=1, stride=1))
        self.pool3 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = torch.nn.Sequential(
            Conv(256, 512, kernel_size=3, padding=1, stride=1),
            Conv(512, 512, kernel_size=3, padding=1, stride=1),
            Conv(512, 512, kernel_size=3, padding=1, stride=1))
        self.pool4 = torch.nn.MaxPool2d(2, 2)

        self.conv5 = torch.nn.Sequential(
            Conv(512, 512, kernel_size=3, padding=1, stride=1),
            Conv(512, 512, kernel_size=3, padding=1, stride=1),
            Conv(512, 512, kernel_size=3, padding=1, stride=1))
        self.pool5 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = Conv(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv7 = Conv(1024, 1024, kernel_size=1, stride=1, padding=0) 
        # padding= 0 중요 ! Conv는 padding default가 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        feat4 = x
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.pool5(x) 
        # (B, 512, 19, 19) here

        x = self.conv6(x)
        feat7 = self.conv7(x)


        return feat4, feat7


class AuxConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv8 = torch.nn.Sequential(
            Conv(1024, 256, kernel_size=1, padding=0),
            Conv(256, 512, kernel_size=3, stride=2, padding=1))

        self.conv9 = torch.nn.Sequential(
            Conv(512, 128, kernel_size=1, padding=0),
            Conv(128, 256, kernel_size=3, stride=2, padding=1))
        
        self.conv10 = torch.nn.Sequential(
            Conv(256, 128, kernel_size=1, padding=0),
            Conv(128, 256, kernel_size=3, padding=0))
        
        self.conv11 = torch.nn.Sequential(
            Conv(256, 128, kernel_size=1, padding=0),
            Conv(128, 256, kernel_size=3, padding=0))
            
    def forward(self, feat7):
        x = self.conv8(feat7)
        feat8 = x

        x = self.conv9(x)
        feat9 = x

        x = self.conv10(x)
        feat10 = x

        x = self.conv11(x)
        feat11 = x

        return feat8, feat9, feat10, feat11
        

class PredConv(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        num_boxes = {
            "feat4":4,
            "feat7":6,
            "feat8":6,
            "feat9":6,
            "feat10":4,
            "feat11":4,
        }
        num_filters = {
            "feat4":512,
            "feat7":1024,
            "feat8":512,
            "feat9":256,
            "feat10":256,
            "feat11":256,
        }

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc4 = torch.nn.Conv2d(num_filters["feat4"], num_boxes["feat4"]*4, kernel_size=3, padding=1)
        self.loc7 = torch.nn.Conv2d(num_filters["feat7"], num_boxes["feat7"]*4, kernel_size=3, padding=1)
        self.loc8 = torch.nn.Conv2d(num_filters["feat8"], num_boxes["feat8"]*4, kernel_size=3, padding=1)
        self.loc9 = torch.nn.Conv2d(num_filters["feat9"], num_boxes["feat9"]*4, kernel_size=3, padding=1)
        self.loc10 = torch.nn.Conv2d(num_filters["feat10"], num_boxes["feat10"]*4, kernel_size=3, padding=1)
        self.loc11 = torch.nn.Conv2d(num_filters["feat11"], num_boxes["feat11"]*4, kernel_size=3, padding=1)

        
        # Class prediction convolutions (predict classes in localization boxes)
        self.class4 = torch.nn.Conv2d(num_filters["feat4"], num_boxes["feat4"]*num_classes, kernel_size=3, padding=1)
        self.class7 = torch.nn.Conv2d(num_filters["feat7"], num_boxes["feat7"]*num_classes, kernel_size=3, padding=1)
        self.class8 = torch.nn.Conv2d(num_filters["feat8"], num_boxes["feat8"]*num_classes, kernel_size=3, padding=1)
        self.class9 = torch.nn.Conv2d(num_filters["feat9"], num_boxes["feat9"]*num_classes, kernel_size=3, padding=1)
        self.class10 = torch.nn.Conv2d(num_filters["feat10"], num_boxes["feat10"]*num_classes, kernel_size=3, padding=1)
        self.class11 = torch.nn.Conv2d(num_filters["feat11"], num_boxes["feat11"]*num_classes, kernel_size=3, padding=1)

        self.init_conv()

    def init_conv(self):
        for c in self.children():
            if isinstance(c, Conv):
                torch.nn.init.xavier_uniform_(c.conv.weight)
                if c.conv.bias is not None:
                    torch.nn.init.constant_(c.conv.bias, 0.)
        return


    def forward(self, feat4, feat7, feat8, feat9, feat10, feat11):
        B = feat4.size(0)

        loc4 = self.loc4(feat4) # (B, 16, 38, 38)
        loc4 = loc4.permute(0,2,3,1).contiguous() # (B, 38, 38, 16)
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        loc4 = loc4.view(B, -1, 4) # (B, 5776, 4), there are a total 5776 boxes on this feature map


        loc7 = self.loc7(feat7) # (B, 24, 19, 19)
        loc7 = loc7.permute(0,2,3,1).contiguous() # (B, 19, 19, 24)
        loc7 = loc7.view(B, -1, 4) # (B, 2166, 4)

        loc8 = self.loc8(feat8) # (B, 24, 10, 10)
        loc8 = loc8.permute(0,2,3,1).contiguous() # (B, 10, 10, 24)
        loc8 = loc8.view(B, -1, 4) # (B, 600, 4)

        loc9 = self.loc9(feat9) # (B, 24, 5, 5)
        loc9 = loc9.permute(0,2,3,1).contiguous() # (B, 5, 5, 24)
        loc9 = loc9.view(B, -1, 4) # (B, 150, 4)

        loc10 = self.loc10(feat10) # (B, 16, 3, 3)
        loc10 = loc10.permute(0,2,3,1).contiguous() # (B, 3, 3, 16)
        loc10 = loc10.view(B, -1, 4) # (B, 36, 4)

        loc11 = self.loc11(feat11) # (B, 16, 1, 1)
        loc11 = loc11.permute(0,2,3,1).contiguous() # (B, 1, 1, 16)
        loc11 = loc11.view(B, -1, 4) # (B, 4, 4)


        # predict classes in localization boxes
        class4 = self.class4(feat4) # (B, 4*nc, 38, 38)
        class4 = class4.permute(0,2,3,1).contiguous() # (B, 38, 38, 4*nc)
        class4 = class4.view(B, -1, self.num_classes) # (B, 5776, nc)

        class7 = self.class7(feat7) # (B, 6*nc, 19, 19)
        class7 = class7.permute(0,2,3,1).contiguous() # (B, 19, 19, 6*nc)
        class7 = class7.view(B, -1, self.num_classes) # (B, 2166, nc)

        class8 = self.class8(feat8) # (B, 6*nc, 10, 10)
        class8 = class8.permute(0,2,3,1).contiguous() # (B, 10, 10, 6*nc)
        class8 = class8.view(B, -1, self.num_classes) # (B, 600, nc)

        class9 = self.class9(feat9) # (B, 6*nc, 5, 5)
        class9 = class9.permute(0,2,3,1).contiguous() # (B, 5, 5, 6*nc)
        class9 = class9.view(B, -1, self.num_classes) # (B, 150, nc)

        class10 = self.class10(feat10) # (B, 4*nc, 3, 3)
        class10 = class10.permute(0,2,3,1).contiguous() # (B, 3, 3, 4*nc)
        class10 = class10.view(B, -1, self.num_classes) # (B, 36, nc)

        class11 = self.class11(feat11) # (B, 4*nc, 1, 1)
        class11 = class11.permute(0,2,3,1).contiguous() # (B, 1, 1, 4*nc)
        class11 = class11.view(B, -1, self.num_classes) # (B, 4, nc)

        locs = torch.cat([loc4, loc7, loc8, loc9, loc10, loc11], dim=1) # (B, 8732, 4)
        classes = torch.cat([class4, class7, class8, class9, class10, class11], dim=1) # (B, 8732, num_classes) 

        return locs, classes



class SSD300(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = None

        self.vgg = VGG()
        self.aux = AuxConv()
        self.pred = PredConv(num_classes=num_classes)

        self.rescale_factors = torch.nn.Parameter(torch.FloatTensor(1, 512, 1, 1)) # 512 channels in conv4 features
        torch.nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()


    def forward(self, x):
        
        # VGG
        feat4, feat7 = self.vgg(x) # (B, 512, 38, 38) | (B, 1024, 19, 19)
        norm = feat4.pow(2).sum(dim=1, keepdim=True).sqrt() # (B, 1, 38, 38)
        feat4 = feat4/norm # (B, 512, 38, 38)
        feat4 = feat4*self.rescale_factors # (B, 512, 38, 38)

        # AUX CONV
        feat8, feat9, feat10, feat11 = self.aux(feat7)
        # (B, 512, 10, 10), 
        # (B, 256, 5, 5),
        # (B, 256, 3, 3),
        # (B, 256, 1, 1),

        locs, classes = self.pred(
            feat4, 
            feat7,
            feat8, 
            feat9,
            feat10,
            feat11,
        )

        return locs, classes


    def create_prior_boxes(self):
        fmap_dims = {
            "feat4":38,
            "feat7":19,
            "feat8":10,
            "feat9":5,
            "feat10":3,
            "feat11":1,
        }

        obj_scales = {
            "feat4":0.1,
            "feat7":0.2,
            "feat8":0.375,
            "feat9":0.55,
            "feat10":0.725,
            "feat11":0.9,
        }

        aspect_ratios = {
            "feat4":[1.0, 2.0, 0.5],
            "feat7":[1.0, 2.0, 3.0, 0.5, 0.333],
            "feat8":[1.0, 2.0, 3.0, 0.5, 0.333],
            "feat9":[1.0, 2.0, 3.0, 0.5, 0.333],
            "feat10":[1.0, 2.0, 0.5],
            "feat11":[1.0, 2.0, 0.5],
        }
        fmap_names = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap_name in enumerate(fmap_names): # index, "feat4"
            for y_ in range(fmap_dims[fmap_name]): # "feat4"-> 38
                for x_ in range(fmap_dims[fmap_name]): # "feat4" -> 38  (feature map sizes, y, x)
                    cx = (x_ + 0.5)/fmap_dims[fmap_name]
                    cy = (y_ + 0.5)/fmap_dims[fmap_name]

                    for ratio in aspect_ratios[fmap_name]: # "feat4" -> ratio : 1.0, 2.0, 0.5
                        # prior boxes
                        prior_boxes.append([
                            cx, 
                            cy, 
                            obj_scales[fmap_name]*sqrt(ratio), 
                            obj_scales[fmap_name]/sqrt(ratio)
                        ])

                        # additional prior boxes
                        # if ratio == 1.0:
                        #     if k+1 < len(fmap_names):
                        #         additional_scale = sqrt(
                        #             obj_scales[fmap_names[k]] # "feat4" -> "feat7"'s obj_scales
                        #             * obj_scales[fmap_names[k+1]] # e.g., 0.1(feat4) -> 0.2(feat7)
                        #         )
                        #     else:
                        #         additional_scale = 1.0                             

                        # prior_boxes.append([
                        #     cx, 
                        #     cy,
                        #     additional_scale, additional_scale
                        # ])
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap_name] * obj_scales[fmap_names[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        # clamp prior boxes
        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device)
        prior_boxes.clamp_(0,1) # (8732, 4)

        return prior_boxes



    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        B = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        
        # (B, 8732, n_classes)
        predicted_scores = torch.nn.functional.softmax(predicted_scores, dim=2)

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)


        # process by iterating single image in B
        for i in range(B):
            # output : (8732, 4) fractional tensor coordinates, (c_x, c_y, w, h)
            decoded_locs = detection_utils.gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy) 
            
            # output : (8732, 4), tensor (x_min, y_min, x_max, y_max)
            decoded_locs = detection_utils.cxcy_to_xy(decoded_locs)

            # lists to store boxes and scores for THIS image
            current_boxes = list()
            current_labels = list()
            current_scores = list()

            # "best" class per each prior box
            max_scores, best_label = predicted_scores[i].max(dim=1) # (8732) <- (8732, n_classes)

            # check for each class
            for c in range(1, self.num_classes):
                class_scores = predicted_scores[i][:, c] # (8732) <- (8732, n_classes), select 'c'
                score_above_min_score = class_scores > min_score # min score : thresholding
                # score_above_min_score : tensor that contains "indices" (n_qualified) 

                n_above_min_score = score_above_min_score.sum().item() # n_qualified list?

                if n_above_min_score == 0:
                    continue # background

                class_scores = class_scores[score_above_min_score] # (n_qualified) | n_min_score <= 8732

                class_decoded_locs = decoded_locs[score_above_min_score] # (n_qualified, 4)


                # Sort predicted boxes and scores by "scores"
                class_scores, sort_idx = class_scores.sort(dim=0, descending=True) #(n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_idx] #(n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = detection_utils.find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                """Non-Maximum Suppression (NMS)"""
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device) #(n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # if this box is already marked for suppression
                    if suppress[box]==1:
                        continue
        
                    # find such boxes whose overlaps (wtih 'this' box) are 
                    # greater than maximum overlap
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # the max operation retains previously suppressed box, like an "OR" operation
                    
                    # don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0
                    
                # Store only unsuppressed boxes for this class
                current_boxes.append(class_decoded_locs[1-suppress])
                current_labels.append(torch.LongTensor((1-suppress).sum().item()*[c]).to(self.device))
                current_scores.append(class_scores[1-suppress])
                    
            # if no object in any class is found, store a placeholder for 'background'
            if len(current_boxes) == 0:
                current_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                current_labels.append(torch.LongTensor([0]).to(self.device))
                current_scores.append(torch.FloatTensor([0.]).to(self.device))

            # concatenate into single tensors
            current_boxes = torch.cat(current_boxes, dim=0) # (n_objects, 4)
            current_labels = torch.cat(current_labels, dim=0) # (n_objects)
            current_scores = torch.cat(current_scores, dim=0) # (n_objects)
            n_objects = current_scores.size(0)

            if n_objects > top_k:
                current_scores, sort_idx = current_scores.sort(dim=0, descending=True)
                current_scores = current_scores[:top_k] # (top_k)
                current_boxes = current_boxes[sort_idx][:top_k] # (top_k, 4)
                current_labels = current_labels[sort_idx][:top_k] # (top_k)

            # append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(current_boxes)
            all_images_labels.append(current_labels)
            all_images_scores.append(current_scores)
        
        return all_images_boxes, all_iamges_labels, all_images_scores
        








