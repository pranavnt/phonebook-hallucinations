from torch.utils.data import Dataset
from faker import Faker
import random
import pickle

class FakePhoneDataset(Dataset):
    def __init__(self, num_people=500, num_samples=10000):
        self.people = {}
        self.num_people = num_people
        self.num_samples = num_samples

        for _ in range(num_people):
            fake = Faker()
            name = fake.name()
            phone_number = fake_phone()
            self.people[name] = phone_number

        self.data = []

        # data entry format;
        # Phonebook:
        # 10 people's names: their phone numbers
        # /n
        # Recall:
        # random name mentioned above: corresponding phone number

        # data is a bunch of strings of this format

        for _ in range(num_samples):
            phonebook = "Phonebook:\n"
            to_recall = "Recall:\n"
            rand_name = random.randint(0, 9)
            ppl_selected = random.sample(list(self.people.keys()), 10)
            for i, name in enumerate(ppl_selected):
                phone_number = self.people[name]
                phonebook += f"{name}: {phone_number}\n"
                if rand_name == i:
                    to_recall += f"{name}: {phone_number}\n"

            self.data.append(phonebook + "\n" + to_recall)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def fake_phone():
    return f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"

if __name__ == "__main__":
    dataset = FakePhoneDataset(2000, 20000)
    with open("phonebook.pkl", "wb") as f:
        pickle.dump(dataset, f)

    ds = pickle.load(open("phonebook.pkl", "rb"))
    print(ds[0])

