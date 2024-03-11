import pickle

import click

from ultralytics import settings, YOLO

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='Enable debug mode', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug
    invoked_subcommand = ctx.invoked_subcommand
    if invoked_subcommand is None:
        logger.info('No subcommand was specified')
    else:
        logger.info(f'Invoked subcommand: {invoked_subcommand}')
        
@router_cmd.command()
@click.option('--path2data', help='Path to source data', default='data/')
@click.option('--path2results', help='Path to output model', default='results/')
def setup(path2data, path2results):
    logger.info('Setting up ultralytics...')
    if not os.path.exists(path2data):
        os.makedirs(path2data)
    if not os.path.exists(path2results):
        os.makedirs(path2results)
        
    settings.update({
        'datasets_dir': path2data,
        'runs_dir': path2results,
        'weights_dir': path2results,
    })
        
@router_cmd.command()
@click.option('--path2test', help='Path to test data', default='data/test/images')
@click.option('--path2output', help='Path to output predictions', default='results/default/')
def default_inference(path2test, path2output):
    logger.info('... [ Predicting with pretrained Yolov8 ] ...')
    if not os.path.exists(path2output):
        os.makedirs(path2output)
        
    model = YOLO('yolov8n.pt')

    images_paths = pull_files(path2test)
    images_names = [os.path.basename(image_path) for image_path in images_paths]
    results = model(images_paths)
    for i, result in enumerate(results):
        # boxes = result.boxes
        # masks = result.masks
        # keypoints = result.keypoints
        # probs = result.probs
        result.save(filename=os.path.join(path2output, images_names[i]))
    
@router_cmd.command()
@click.option('--yaml_file', help='Path to train yaml file', default='data/data.yaml')
@click.option('--num_epochs', help='Number of epochs', default=5)
def train(yaml_file, num_epochs):
    logger.debug('Training...')
    model = YOLO('yolov8n.pt')
    model.train(data=yaml_file, epochs=num_epochs)
    
@router_cmd.command()
@click.option('--path2model', help='path to model', type=click.Path(True), default='results/detect/train/weights/best.pt')
@click.option('--path2test', help='Path to test data', default='data/test/images')
def ft_inference(path2model, path2test):
    logger.debug('Inference...')
    predictor = YOLO(path2model)
    predictor.predict(path2test, save = True, save_txt = True)
        
        
if __name__ == '__main__':
    logger.info('...')
    router_cmd(obj={})