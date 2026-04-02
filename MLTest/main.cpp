// main.cpp
#include <iostream>
#include <mlcore/memory/storage.h>

int main() {
    MLCore::Memory::ArenaAllocator arena(1024);

    auto storage = MLCore::Memory::MakeStorage<float>(arena, 10);

    for (int i = 0; i <= 10; i++) {
        storage.Data()[i] = i * 2.0f;
    }

    for (int i = 0; i <= 10; i++) {
        std::cout << storage.Data()[i] << "\n";
    }
}