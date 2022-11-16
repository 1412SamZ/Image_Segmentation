# This repo will contain UNet ... and other models!
Nowadays, a huge, complex model consists of several basic models like CNN, FCN, attention...
So this repo is made for myself to learn those SOTA-models and implement them from scratch, instead of import from somewhere

# Hardware
Running on my Legion laptop
I will run all those experiments with this setup.
- CPU: 8th-gen i5
- GPU: GTX1060-6GB
- MEM: 16GB




# U-Net

![U-NET](u-net-architecture.png)

- Image credit to [Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.]


# Others
- Helped me a lot by the problem: committed the state_dict which is over 100MiB large and cannot push my repo, and then also cannot delete it from the git objects
- Solution: git reset HEAD~5 (5 is from my example, check it by using git status and it returns -> Your branch is ahead of 'origin/main' by 5 commits.)
- Then ignore those large files and add-commit-push!