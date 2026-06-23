# Uses of the Masked Autoencoder Model in Later Literature: Expanded Full-Text Review

He et al. introduced masked autoencoders (MAE) as a scalable self-supervised framework for vision representation learning, built around random patch masking, an asymmetric encoder-decoder architecture, and pixel-level reconstruction of missing image patches [1]. The encoder processes only visible patches, the decoder reconstructs the full image from latent representations and mask tokens, and the decoder is typically discarded after pretraining. The important methodological insight was that high masking ratios can turn visually redundant images into a difficult but informative reconstruction task, while reducing encoder computation and improving transfer to downstream recognition tasks.

The later literature shows that MAE has become both a concrete pretrained model family and a broader design pattern for self-supervised representation learning. In the expanded local full-text pass, 42 extracted papers in `open-pdfs-text` were inspected for explicit uses of "Masked Autoencoders Are Scalable Vision Learners," arXiv:2111.06377, MAE, ViT-MAE, VideoMAE, masked image modeling, masked reconstruction, high masking ratios, and MAE-derived benchmarking. The evidence varies in strength. Some papers directly use MAE pretrained weights or implement a MAE-like encoder-decoder objective. Others adapt only a component, such as random masking. A third group cites MAE as background or as a future direction; those papers are not treated here as primary examples of use.

One early and direct form of use is benchmarking MAE as a transferable vision initialization. In detection transfer learning, Li et al. used the ViT-B and ViT-L weights pretrained by the MAE authors on unsupervised ImageNet-1K and compared them with BEiT, MoCo v3, supervised ImageNet initialization, and random initialization for COCO detection and segmentation [2]. The study reported that masking-based methods, especially MAE and BEiT, provided convincing improvements in AP and faster convergence relative to random initialization, with stronger gains at larger ViT scale. In this sense, MAE is used not as a new architectural component, but as a pretrained representation whose downstream transfer behavior is experimentally evaluated.

Other computer-vision papers use MAE as a methodological baseline for improving masked image modeling itself. PeCo proposes a perceptual codebook for BERT-style pretraining of vision transformers and repeatedly compares its transfer results with MAE, including image classification, semantic segmentation, and object detection [3]. The paper also discusses MAE's asymmetric encoder-decoder design and reports that perceptual prediction targets can improve downstream performance relative to raw-pixel MAE targets in some settings. MC-SSL0.0 similarly builds on group masked modeling and masked-autoencoder ideas, combining patch classification and patch reconstruction losses, discarding task heads after pretraining, and fine-tuning the pretrained backbone for downstream multi-label and multi-class tasks [4]. These works show MAE functioning as a reference point for the design of better masking targets and richer self-supervised objectives.

Histopathology is a major biomedical setting in which MAE-like masked image modeling has been used as a scalable pretraining paradigm. Filiot et al. review MAE as the work that allowed ViTs to be used effectively for masked image modeling, then evaluate self-supervised ViT representations for histology across patch-level and slide-level tasks [5]. The paper is not a simple reuse of the original MAE; rather, it treats MAE as part of a family of MIM methods, including BEiT, SimMIM, and iBOT, that can be scaled for whole-slide pathology representation learning. Related healthcare benchmarking work evaluates large collections of embedding models, including ViT models trained with Masked Autoencoders, DINO, CLIP, and supervised objectives, on dermatology classification and bias analyses [6]. In these studies, MAE is important as both a pretraining strategy and a comparator within broader foundation-model evaluation.

A prominent line of later work extends MAE from still images to video and spatiotemporal data. BEVT performs masked image modeling on image data and then jointly conducts masked image and masked video modeling on video data, using image pretraining as a spatial prior for video transformers [7]. AV-MaskEnhancer builds on masked autoencoding for video representation learning but argues that visual-only reconstruction can be limited in low-quality videos, motivating the incorporation of audio-visual information [8]. Echo-Vision-FM provides a domain-specific medical example: it uses Echo-VideoMAE, a VideoMAE-derived autoencoder with non-overlapping video patches, high-ratio tube masking, and an asymmetric encoder-decoder architecture, to pretrain on echocardiogram videos before fine-tuning for downstream cardiac tasks [9]. Together, these papers preserve the MAE logic of reconstructive pretraining while changing the input unit from image patches to video tubes or multimodal video features.

Retinal and brain MRI foundation models provide further evidence that the MAE template has become attractive in medical imaging, where labeled data are expensive but unlabeled images are abundant. SLOTMFound uses RETFound-Fundus as a backbone and describes RETFound as a vision-transformer retinal model pretrained with masked autoencoding; it then adapts the foundation model to scanning laser ophthalmoscopy images and OCT-derived thickness maps, using masked image modeling to reconstruct missing regions and obtain SLOFound and TMFound encoders for multiple-sclerosis classification [10]. BrainMRIFM develops slice and volume brain MRI foundation models and explicitly includes an MAE-ViT slice foundation model among the pretrained variants evaluated across downstream brain tumor tasks [11]. These examples are more specialized than the original ImageNet MAE, but they retain the same basic assumption: reconstruction of masked image structure can yield transferable encoders for clinical downstream models.

MAE-style learning has also been translated to 3D and structured scientific inputs. Point-BERT extends masked modeling to point clouds, citing MAE as a recent masked-autoencoder strategy and then using masked point modeling with block-wise masking and discrete point tokens to infer missing geometric structure [12]. A masked-autoencoder-based three-dimensional foundation model for vortex identification applies MAE pretraining to 3D PDE-derived flow-field data, uses high masking ratios to learn latent representations, and fine-tunes the pretrained model for sparse-data vortex identification across multiple PDE datasets [13]. In structural immunology, work on MHC-bound peptide prediction cites MAE as an example of self-supervised masking and extends the idea to 3D protein complexes through masked residue prediction, where residues are hidden and an equivariant graph neural network learns to recover masked residue identities before downstream binding prediction [14]. Environmental sensing work similarly trains a transformer-based masked autoencoder to reconstruct pollutant and meteorological fields from sparse mobile measurements, with synthetic pretraining followed by fine-tuning on field surveys [15].

Scientific imaging and physical measurement systems often modify MAE to encode prior structure or domain physics. PiMAE introduces a physics-informed masked autoencoder for optical microscopy, using self-supervised reconstruction to estimate the point spread function and emitters directly from raw microscopy images rather than requiring a known PSF [16]. Quaternion-based ViT-MAE adapts masked autoencoding to polycrystalline EBSD orientation maps by representing crystal orientations with quaternions and pretraining on synthetic data before testing transfer to real scans [17]. Remote-sensing papers use masked image modeling to pretrain ViT backbones for scene classification and Earth-observation representation learning, while agricultural image classification work applies the same pretraining-fine-tuning logic to agricultural image corpora [18-20]. In these domains, the MAE contribution is less the specific ImageNet model than the reusable masked reconstruction recipe.

The MAE idea has moved beyond conventional imagery into multimodal, omics, neural, and behavioral data. scMMAE extends masked autoencoding with cross-attention for single-cell multimodal omics fusion, learning shared and modality-specific information from transcriptomic and proteomic data and transferring the learned representation to unimodal omics analysis [21]. A later robust multi-scale single-cell clustering framework uses scMAE as an important masked-autoencoder comparator and reports improved clustering metrics relative to scMAE and other deep clustering baselines [22]. RamanMAE uses masked autoencoders to learn biologically meaningful spectral representations for Raman molecular imaging [23]. fNIRS foundation-model work, SeqMAE for EEG-derived brain activity decoding, and hBehaveMAE for motion-capture behavior analysis further illustrate the abstraction of MAE away from RGB image patches toward general masked reconstruction over structured biological or behavioral signals [24-26]. Synthetic solar irradiance work likewise includes masked-autoencoder-based generative models for sequence construction and downstream forecasting or planning evaluations [27].

Several papers use MAE as a component in hybrid applied systems or adapt one of its design mechanisms. MAE-based image inpainting-steganography explicitly proposes an MAE-based framework for damaged color images, using non-overlapping patches, random masking, an asymmetric encoder-decoder reconstruction mechanism, and a feature-domain embedding strategy to combine image inpainting with robust secret-message embedding [28]. In robotics, Voltron contrasts masked autoencoding with contrastive learning, arguing that MAE-like methods tend to preserve low-level spatial information while contrastive methods emphasize higher-level semantics; its language-driven visual reconstruction objective can therefore be read as a response to MAE's representational strengths and weaknesses [29]. In sign-language recognition and translation, random visual masking is explicitly added as a hard augmentation inspired by MAE, improving visual representation learning under a contrastive training objective [30]. In incomplete multi-view partial multi-label classification, VLCSA introduces a view-label hybrid-driven autoencoder and a stochastic masking strategy inspired by He et al., using random feature suppression to reduce redundancy and support missing-view completion [31]. These works do not always reproduce the full MAE architecture, but they show how MAE's masking principle has become a portable design motif.

The local full-text pass also identified papers that cite MAE but should be treated cautiously. Surveys on attention mechanisms and cross-modal representation learning discuss MAE as part of the transformer and masked-patch-pretraining landscape, but they are background sources rather than downstream uses. PillarNeSt cites MAE and discusses masked image modeling as a possible future generative pretraining strategy for 2D pseudo-images in pillar-based 3D detection, but the implemented method uses ConvNeXt-style image-domain pretraining rather than MAE [32]. BrainIAC cites MAE as a possible future SSL framework for brain MRI but uses SimCLR-style contrastive pretraining [33]. Vision Transformer Autoencoders for genetic association analysis cites MAE and reconstructs brain MRI inputs with a ViT autoencoder, but the local evidence supports it as MAE-adjacent rather than a direct MAE implementation [34]. Other papers, including work on canopy-height mapping, malaria diagnosis, typography, and music composer recognition, cite MAE or ViT-MAE as motivation or future opportunity without clear full-text evidence that MAE was implemented or benchmarked as part of the reported method [35].

Overall, later papers use MAE in four scientifically distinct ways. First, they directly evaluate MAE pretrained representations as transferable backbones, especially in detection, segmentation, dermatology, and brain MRI benchmarking. Second, they modify the masked image modeling objective itself, changing the prediction target, masking strategy, or reconstruction unit. Third, they transfer the masked reconstruction principle to new data structures, including videos, point clouds, PDE fields, protein structures, environmental maps, spectra, omics profiles, EEG, fNIRS, behavior trajectories, and retinal or MRI images. Fourth, they use MAE as a conceptual reference for understanding tradeoffs between reconstruction-based, contrastive, and multimodal pretraining. The strongest evidence of use comes from papers that explicitly instantiate masked autoencoders, MAE-ViT, VideoMAE, ViT-MAE, scMMAE, RamanMAE, PiMAE, or MAE-based reconstruction frameworks, or that use the MAE authors' pretrained weights in downstream experiments.

## Limitations

This expanded review combines evidence from the earlier citation-classification review with a new inspection of the 42 extracted full-text records in `open-pdfs-text`. The local pass improves coverage of papers that were available as PDFs in the MAE folder, but it does not constitute a complete review of all papers citing arXiv:2111.06377. Some references were available only through the earlier review evidence rather than local full text. Conversely, several local full texts cited MAE only in the introduction, related work, references, or future-work discussion; these were excluded from the main synthesis unless they implemented, adapted, or benchmarked MAE-style mechanisms. Claims about performance are reported cautiously and at a high level because this review did not reanalyze numerical tables or reproduce experiments.

## References

[1] He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, R. Masked Autoencoders Are Scalable Vision Learners. arXiv:2111.06377, 2021. DOI: 10.48550/arXiv.2111.06377.

[2] Benchmarking Detection Transfer Learning with Vision Transformers. DOI: 10.48550/arXiv.2111.11429.

[3] PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers. DOI: 10.48550/arXiv.2111.12710.

[4] MC-SSL0.0: Towards Multi-Concept Self-Supervised Learning. DOI: 10.48550/arXiv.2111.15340.

[5] Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling. DOI: 10.1101/2023.07.21.23292757.

[6] A Framework for Evaluating the Efficacy of Foundation Embedding Models in Healthcare. DOI: 10.1101/2024.04.17.24305983.

[7] BEVT: BERT Pretraining of Video Transformers. DOI: 10.1109/CVPR52688.2022.01432.

[8] AV-MaskEnhancer: Enhancing Video Representations through Audio-Visual Masked Autoencoder. DOI: 10.1109/ICTAI59109.2023.00058.

[9] Echo-Vision-FM: A Pre-training and Fine-tuning Framework for Echocardiogram Video Vision Foundation Model. DOI: 10.1101/2024.10.09.24315195.

[10] SLOTMFound: Foundation-Based Diagnosis of Multiple Sclerosis Using Retinal SLO Imaging and OCT Thickness-maps. DOI: 10.1101/2025.07.14.25331522.

[11] From Slices to Volumes: A Scalable Pipeline for Developing General-Purpose Brain MRI Foundation Models. DOI: 10.1101/2025.04.12.25325728.

[12] Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling. DOI: 10.48550/arXiv.2111.14819.

[13] A masked autoencoder-based three-dimensional foundation model for vortex identification. DOI: 10.1063/5.0281019.

[14] Geometric deep learning improves generalizability of MHC-bound peptide predictions. DOI: 10.1038/s42003-024-07292-1.

[15] Learning Neighborhood-Scale Cross-Dependencies Among Air Pollutants, Meteorology and Land Cover Using Mobile Sensing and Transformers. DOI: 10.21203/rs.3.rs-8011461/v1.

[16] Learning the imaging mechanism directly from optical microscopy observations. DOI: 10.1364/PRJ.488310.

[17] Quaternion-based vision-transformer for polycrystalline EBSD scans pre-trained on large-scale synthetic data. DOI: 10.1016/j.matdes.2025.114599.

[18] Remote sensing scene classification with masked image modeling. DOI: 10.1117/12.2680898.

[19] Self-supervised Pretraining of Vision Transformers for Earth Observation. DOI: 10.22215/etd/2023-15793.

[20] Optimizing agricultural classification with masked image modeling. DOI: 10.1080/23311932.2025.2462243.

[21] scMMAE: masked cross-attention network for single-cell multimodal omics fusion to enhance unimodal omics. DOI: 10.1093/bib/bbaf010.

[22] A robust multi-scale clustering framework for single-cell RNA-seq data analysis. DOI: 10.1038/s41598-025-03603-6.

[23] RamanMAE: Masked Autoencoders Enable Efficient Molecular Imaging by Learning Biologically Meaningful Spectral Representations. DOI: 10.1101/2025.05.18.654618.

[24] fNIRS Foundation Model for Few-Shot Based fNIRS Classification. DOI: 10.1109/BCI65088.2025.10931275.

[25] Robust Representation from EEG via Pre-trained SeqMAE for Brain Activity Decoding. DOI: 10.1109/IJCNN64981.2025.11228525.

[26] Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders. DOI: 10.1101/2024.08.06.606796.

[27] Generative neural network models for synthetic solar irradiance sequences. DOI: 10.1063/5.0219923.

[28] MAE-based image inpainting-steganography method. DOI: 10.1186/s42400-025-00486-y.

[29] Language-Driven Representation Learning for Robotics. DOI: 10.15607/RSS.2023.XIX.032.

[30] Contrastive Learning for Sign Language Recognition and Translation. DOI: 10.24963/IJCAI.2023/85.

[31] View-label driven cross-space structure alignment for incomplete multi-view partial multi-label classification. DOI: 10.1007/s44443-025-00130-2.

[32] PillarNeSt: Embracing Backbone Scaling and Pretraining for Pillar-Based 3D Object Detection. DOI: 10.1109/TIV.2024.3386576.

[33] A foundation model for generalized brain MRI analysis. DOI: 10.1101/2024.12.02.24317992.

[34] Vision Transformer Autoencoders for Unsupervised Representation Learning: Revealing Novel Genetic Associations through Learned Sparse Attention Patterns. DOI: 10.1101/2025.03.24.25324549.

[35] PBSCR: The Piano Bootleg Score Composer Recognition Dataset. DOI: 10.5334/tismir.185.
