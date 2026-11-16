package dev.seedwright.oracle;

import com.mojang.datafixers.util.Pair;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.Climate;
import net.minecraft.world.level.biome.MultiNoiseBiomeSource;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGeneratorStructureState;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.structure.Structure;
import net.minecraft.world.level.levelgen.structure.StructureSet;
import net.minecraft.world.level.levelgen.structure.StructureStart;
import net.minecraft.world.level.levelgen.structure.placement.ConcentricRingsStructurePlacement;

/**
 * Ground-truth dumper for the legacy (double precision) worldgen API, MC 1.21.1 - 26.2. Same output format
 * as the modern oracle minus the column section (legacy has no separate volume path). Only API that exists
 * unchanged across all those versions is used; drifting bits go through reflection or toString.
 */
public final class Oracle
{
	private Oracle() {}

	/** "ResourceKey[minecraft:worldgen/biome / minecraft:plains]" -> "minecraft:plains" (no ResourceLocation/Identifier). */
	private static String id(Holder<?> h)
	{
		return h.unwrapKey().map(k -> {
			String s = k.toString();
			return s.substring(s.lastIndexOf(" / ") + 3, s.length() - 1);
		}).orElse("?");
	}

	private static int cx(ChunkPos p) { return p.getMinBlockX() >> 4; }
	private static int cz(ChunkPos p) { return p.getMinBlockZ() >> 4; }

	public static void run(MinecraftServer server)
	{
		try
		{
			dump(server);
		}
		catch (Throwable t)
		{
			t.printStackTrace();
		}
		server.halt(false);
	}

	private static BlockPos spawnPos(ServerLevel level) throws Exception
	{
		for (String name : new String[] {"getSharedSpawnPos"})
		{
			try
			{
				return (BlockPos) level.getClass().getMethod(name).invoke(level);
			}
			catch (NoSuchMethodException ignored) {}
		}
		Object data = level.getClass().getMethod("getRespawnData").invoke(level);
		return (BlockPos) data.getClass().getMethod("pos").invoke(data);
	}

	@SuppressWarnings("unchecked")
	private static void dump(MinecraftServer server) throws Exception
	{
		ServerLevel level = server.overworld();
		long seed = level.getSeed();
		var generator = level.getChunkSource().getGenerator();
		RandomState randomState = level.getChunkSource().randomState();
		BiomeSource biomeSource = generator.getBiomeSource();
		StringBuilder out = new StringBuilder();

		Method params = MultiNoiseBiomeSource.class.getDeclaredMethod("parameters");
		params.setAccessible(true);
		Climate.ParameterList<Holder<Biome>> list = (Climate.ParameterList<Holder<Biome>>) params.invoke(biomeSource);
		StringBuilder bp = new StringBuilder("# biome tmin tmax hmin hmax cmin cmax emin emax dmin dmax wmin wmax offset\n");
		for (Pair<Climate.ParameterPoint, Holder<Biome>> p : list.values())
		{
			Climate.ParameterPoint q = p.getFirst();
			bp.append(id(p.getSecond()));
			for (Climate.Parameter r : List.of(q.temperature(), q.humidity(), q.continentalness(), q.erosion(), q.depth(), q.weirdness()))
			{
				bp.append(' ').append(r.min()).append(' ').append(r.max());
			}
			bp.append(' ').append(q.offset()).append('\n');
		}
		Files.writeString(Path.of("biome_parameters.txt"), bp);

		int points = Integer.getInteger("seedwright.oracle.points", 4000);
		int radius = Integer.getInteger("seedwright.oracle.radius", 40);
		Random rnd = new Random(seed ^ 0x5eed);
		out.append("seed ").append(seed).append('\n');

		Climate.Sampler sampler = randomState.sampler();
		out.append("points ").append(points).append('\n');
		for (int i = 0; i < points; i++)
		{
			int qx = rnd.nextInt(10000) - 5000;
			int qy = rnd.nextInt(96) - 16;
			int qz = rnd.nextInt(10000) - 5000;
			if (i % 4 == 0)
			{
				qx = rnd.nextInt(1_000_000) - 500_000;
				qz = rnd.nextInt(1_000_000) - 500_000;
			}
			var ctx = new DensityFunction.SinglePointContext(qx << 2, qy << 2, qz << 2);
			float[] v = {
				(float) sampler.temperature().compute(ctx), (float) sampler.humidity().compute(ctx), (float) sampler.continentalness().compute(ctx),
				(float) sampler.erosion().compute(ctx), (float) sampler.depth().compute(ctx), (float) sampler.weirdness().compute(ctx)
			};
			out.append(qx).append(' ').append(qy).append(' ').append(qz);
			for (float f : v)
			{
				out.append(' ').append(Integer.toHexString(Float.floatToRawIntBits(f)));
			}
			out.append(' ').append(id(biomeSource.getNoiseBiome(qx, qy, qz, sampler))).append('\n');
		}
		out.append("columns 0\n");

		BlockPos origin = sampler.findSpawnPosition();
		BlockPos spawn = spawnPos(level);
		out.append("spawn_origin ").append(origin.getX() >> 4).append(' ').append(origin.getZ() >> 4).append('\n');
		out.append("spawn ").append(spawn.getX()).append(' ').append(spawn.getY()).append(' ').append(spawn.getZ()).append('\n');

		ChunkGeneratorStructureState state = level.getChunkSource().getGeneratorState();
		state.ensureStructuresGenerated();
		Map<Structure, String> names = new IdentityHashMap<>();
		StringBuilder cands = new StringBuilder();
		int candCount = 0;
		for (Holder<StructureSet> set : state.possibleStructureSets())
		{
			for (StructureSet.StructureSelectionEntry e : set.value().structures())
			{
				names.put(e.structure().value(), id(e.structure()));
			}
			if (set.value().placement() instanceof ConcentricRingsStructurePlacement rings)
			{
				List<ChunkPos> ring = state.getRingPositionsFor(rings);
				out.append("rings ").append(id(set)).append(' ').append(ring.size());
				for (ChunkPos p : ring)
				{
					out.append(' ').append(cx(p)).append(' ').append(cz(p));
				}
				out.append('\n');
			}
			for (int x = -radius; x <= radius; x++)
			{
				for (int z = -radius; z <= radius; z++)
				{
					if (set.value().placement().isStructureChunk(state, x, z))
					{
						cands.append(id(set)).append(' ').append(x).append(' ').append(z).append('\n');
						candCount++;
					}
				}
			}
		}
		out.append("candidates ").append(candCount).append('\n').append(cands);

		StringBuilder starts = new StringBuilder();
		int startCount = 0;
		for (int x = -radius; x <= radius; x++)
		{
			for (int z = -radius; z <= radius; z++)
			{
				ChunkAccess chunk = level.getChunk(x, z, ChunkStatus.STRUCTURE_STARTS, true);
				for (Map.Entry<Structure, StructureStart> e : chunk.getAllStarts().entrySet())
				{
					if (!e.getValue().isValid())
					{
						continue;
					}
					var box = e.getValue().getBoundingBox();
					starts.append(names.getOrDefault(e.getKey(), "?")).append(' ').append(x).append(' ').append(z)
						.append(' ').append(box.minX()).append(' ').append(box.minY()).append(' ').append(box.minZ())
						.append(' ').append(box.maxX()).append(' ').append(box.maxY()).append(' ').append(box.maxZ()).append('\n');
					startCount++;
				}
			}
		}
		out.append("starts ").append(startCount).append('\n').append(starts);
		Files.writeString(Path.of("oracle-" + seed + ".txt"), out);

		// ---- jigsaw layouts (villages, outposts) with terrain heights: -Dseedwright.oracle.villages=R ----
		int villages = Integer.getInteger("seedwright.oracle.villages", 0);
		if (villages > 0)
		{
			JigsawDump.dump(level, generator, randomState, state, seed, villages, Oracle::id, names);
		}

		// ---- chest loot (villages, buried treasure): -Dseedwright.oracle.loot=R ----
		int loot = Integer.getInteger("seedwright.oracle.loot", 0);
		if (loot > 0)
		{
			LootDump.dump(level, state, seed, loot, Oracle::id, names);
		}
	}
}
