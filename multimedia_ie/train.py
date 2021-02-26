import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
def fix_seed(config):
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

def get_optimizer(config, models):
    parameters = []
    for model in models:
        parameters += [p for p in model.parameters() if p.requires_grad]

    if config.optimizer == "adam":
        return optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    elif config.optimizer == "adamw":
        return AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    else:
        return optim.SGD(parameters, momentum=0.9, lr=config.learning_rate, weight_decay=config.weight_decay)

def test(text_model, mention_model, prediction_model, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    all_event_preds = []
    all_role_preds = []
    all_entity_preds = []
    all_event_labels = []
    all_role_labels = []
    all_entity_labels = []
    text_model.eval()
    prediction_model.eval()

    with torch.no_grad():
      pred_dicts = []
      for i, batch in enumerate(test_loader):
        doc_embeddings,\
        start_mappings,\
        end_mappings,\
        continuous_mappings,\
        width,\
        event_labels,\
        role_labels,\
        entity_labels,\
        event_mappings,\
        role_mappings,\
        entity_mappings,\
        text_mask, span_mask = batch
        
        doc_embeddings = doc_embeddings.to(device)
        start_mappings = start_mappings.to(device)  
        continuous_mappings = continuous_mappings.to(device)
        width = width.to(device)
        event_mappings = event_mappings.to(device)
        role_mappings = role_mappings.to(device)
        entity_mappings = entity_mappings.to(device)
        event_labels = event_labels.to(device)
        role_labels = role_labels.to(device)
        entity_labels = entity_labels.to(device)

        arg_num = role_mappings.sum(-1).sum(-1).long()
        # Compute mention embeddings
        text_output = text_model(doc_embeddings)
        mention_output = mention_model(text_output, start_mappings, end_mappings, continuous_mappings, width)
        event_embs = torch.matmul(event_mappings, mention_output)
        arg_embs = torch.matmul(role_mappings, mention_output)
        for idx in range(B):
          event_scores, arg_scores, entity_scores = JointClassifier(event_emb, arg_embs[idx, :arg_num[idx]])
          # TODO Save outputs
          all_event_preds.append(event_scores.max(dim=1)[-1])
          all_role_preds.append(arg_scores.max(dim=1)[-1])
          all_entity_preds.append(entity_scores.max(dim=1)[-1])  
          all_event_labels.append(event_labels[idx])
          all_role_labels.append(role_labels[idx, :arg_num[idx]])
          all_entity_labels.append(entity_labels[idx, :arg_num[idx]])

      all_event_preds = torch.cat(all_event_preds)
      all_role_preds = torch.cat(all_role_preds)
      all_entity_preds = torch.cat(all_entity_preds)
      all_event_labels = torch.cat(all_event_labels)
      all_role_labels = torch.cat(all_role_labels)
      all_entity_labels = torch.cat(all_entity_labels) 
      
      event_acc = (all_event_preds == all_event_labels).mean()
      entity_acc = (all_entity_preds == all_entity_labels).mean()
      role_acc = (all_role_preds == all_role_labels).mean()
      
      info = 'Event classification accuracy {:.3f}\tArgument classification accuracy {:.3f}\tEntity classification accuracy {:.3f}'.format(event_acc, role_acc, entity_acc)
      print(info)
      logger.info(info)
    return event_acc, role_acc, entity_acc


def train(text_model, mention_model, prediction_model, train_loader, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  fix_seed(config)

  # for p in text_model.parameters():
  #   print(p.requires_grad)
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(mention_model, torch.nn.DataParallel):
    mention_model = nn.DataParallel(mention_model)

  if not isinstance(prediction_model, torch.nn.DataParallel):
    prediction_model = nn.DataParallel(prediction_model)

  text_model.to(device)
  prediction_model.to(device)

  # Create/load exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)

  # Define the training criterion
  criterion = nn.CrossEntropyLoss()
  
  # Set up the optimizer
  optimizer = get_optimizer(config, [text_model, prediction_model])

  # Start training
  total_loss = 0.
  total = 0.
  begin_time = time.time() 
  
  if args.evaluate_only:
    config.epochs = 0

  best_acc = None
  for epoch in range(args.start_epoch, config.epochs):
    text_model.train()
    image_model.train()
    coref_model.train()
    for i, batch in enumerate(train_loader):
      doc_embeddings,\
      start_mappings,\
      end_mappings,\
      continuous_mappings,\
      width,\
      event_labels,\
      role_labels,\
      entity_labels,\
      event_mappings,\
      role_mappings,\
      entity_mappings,\
      text_mask, span_mask = batch
      
      B = doc_embeddings.size(0)
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)  
      continuous_mappings = continuous_mappings.to(device)
      width = width.to(device)
      event_mappings = event_mappings.to(device)
      role_mappings = role_mappings.to(device)
      entity_mappings = entity_mappings.to(device)
      event_labels = event_labels.to(device)
      role_labels = role_labels.to(device)
      entity_labels = entity_labels.to(device)

      optimizer.zero_grad()
      text_output = text_model(doc_embeddings)
      mention_output = mention_model(text_output, start_mappings, end_mappings, continuous_mappings, width)
      event_embs = torch.matmul(event_mappings, mention_output)
      arg_embs = torch.matmul(role_mappings, mention_output)
      event_scores, arg_scores, entity_scores = JointClassifier(event_emb, arg_embs)
      
      loss = criterion(event_scores, event_labels) +\
             criterion(arg_scores, role_labels) +\
             criterion(entity_scores, entity_labels)

      loss.backward()
      optimizer.step()

      total_loss += loss.item() * B
      total += B
      if i % 50 == 0:
        test(text_model, mention_model, prediction_model, test_loader, args)

    info = 'Epoch: [{}][{}/{}]\tTime {:.3f}\tLoss total {:.4f} ({:.4f})'.format(epoch, i, len(train_loader), time.time()-begin_time, total_loss, total_loss / total)
    print(info)
    logger.info(info)

    if epoch % 1 == 0:
      event_acc, role_acc, entity_acc = test(text_model, mention_model, prediction_model, test_loader, args)
      avg_acc = (event_acc + role_acc + entity_acc) / 3
      if epoch == 0:
        best_acc = avg_acc
      elif avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(text_model.module.state_dict(), '{}/text_model.{}.pth'.format(args.exp_dir, epoch))
        torch.save(mention_model.module.state_dict(), '{}/mention_model.{}.pth')
        torch.save(prediction_model.module.state_dict(), '{}/prediction_model.{}.pth'.format(args.exp_dir, epoch))

  if args.evaluate_only:
    _, _, _ = test(text_model, mention_model, prediction_model, test_loader, args)


if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='')
  parser.add_argument('--config', type=str, default='configs/config_translation.json')
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--evaluate_only', action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Set up logger
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if not args.exp_dir:
    args.exp_dir = config['model_path']
  else:
    config['model_path'] = args.exp_dir
    
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  if not os.path.isdir(config['log_path']):
    os.makedirs(config['log_path']) 
  
  pred_out_dir = os.path.join(config['model_path'], 'pred')
  if not os.path.isdir(pred_out_dir):
    os.mkdir(pred_out_dir)

  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 
  
  # Initialize dataloaders
  preprocessor = Preprocessor(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_events_with_arguments.json') 
  train_set = ACEDataset(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_events_with_arguments.json'), preprocessor, config, split='train')
  test_set = ACEDataset(os.path.join(config['data_folder'], 'test.json'), os.path.join(config['data_folder'], 'test_events_with_arguments.json'), preprocessor, config, split='test')
  
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  text_model = BiLSTM(config.bert_hidden_size, config.bert_hidden_size)
  mention_model = SpanEmbedder(config, device).to(device)
  prediction_model = JointClassifier(preprocessor.n_event_types, preprocessor.n_role_types, preprocessor.n_entity_types, config)
  
  train(text_model, mention_model, prediction_model, train_loader, test_loader, args) 
